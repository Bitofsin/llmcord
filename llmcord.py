import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

import discord
import httpx
from openai import AsyncOpenAI
import yaml

# ——— Logging ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# ——— Constants ———
VISION_MODEL_TAGS = (
    "gpt-4", "o3", "o4", "claude-3",
    "gemini", "gemma", "llama", "pixtral",
    "mistral-small", "vision", "vl",
)
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500

# ——— Config loader ———
def get_config(filename="config.yaml"):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

cfg = get_config()

if client_id := cfg.get("client_id"):
    logging.info(
        f"\n\n BOT INVITE URL:\n"
        f" https://discord.com/api/oauth2/authorize?"
        f"client_id={client_id}&permissions=412317273088&scope=bot\n"
    )

# ——— Discord client setup ———
intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(
    name=(cfg.get("status_message") or "github.com/Bitofsin/llmcord")[:128]
)
discord_client = discord.Client(intents=intents, activity=activity)

# ——— HTTP client & state ———
httpx_client = httpx.AsyncClient()
msg_nodes = {}
last_task_time = 0

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    #— reload config for hot-reload —
    cfg = get_config()
    use_n8n     = cfg.get("use_n8n", False)
    n8n_webhook = cfg.get("n8n_webhook_url", "")

    #— ignore bots, require mention in guilds or allow DMs —
    is_dm = new_msg.channel.type == discord.ChannelType.private
    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    #— permissions check (unchanged) —
    role_ids = {r.id for r in getattr(new_msg.author, "roles", ())}
    channel_ids = set(filter(None, (
        new_msg.channel.id,
        getattr(new_msg.channel, "parent_id", None),
        getattr(new_msg.channel, "category_id", None),
    )))
    allow_dms   = cfg["allow_dms"]
    permissions = cfg["permissions"]
    (allowed_user_ids, blocked_user_ids), \
    (allowed_role_ids, blocked_role_ids), \
    (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"])
        for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user  = allow_all_users or new_msg.author.id in allowed_user_ids or bool(role_ids & set(allowed_role_ids))
    is_bad_user   = not is_good_user or new_msg.author.id in blocked_user_ids or bool(role_ids & set(blocked_role_ids))
    allow_all_channels = not allowed_channel_ids
    is_good_channel   = allow_dms if is_dm else allow_all_channels or bool(channel_ids & set(allowed_channel_ids))
    is_bad_channel    = not is_good_channel or bool(channel_ids & set(blocked_channel_ids))
    if is_bad_user or is_bad_channel:
        return

    # ——— NEW: n8n branch ———
    if use_n8n:
        if not n8n_webhook:
            logging.error("n8n_webhook_url not set in config.yaml")
            return

        payload = {
            "channel_id": new_msg.channel.id,
            "author_id":  new_msg.author.id,
            "content":    new_msg.content,
        }

        resp = await httpx_client.post(n8n_webhook, json=payload)
        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("reply")
            if reply:
                await new_msg.reply(reply)
        else:
            logging.error(f"n8n webhook failed: {resp.status_code} {await resp.text()}")
        return

    # ——— FALLBACK: original OpenAI streaming logic ———

    # LLM provider init
    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key  = cfg["providers"][provider].get("api_key", "")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # per-model/image settings
    accept_images    = any(tag in model.lower() for tag in VISION_MODEL_TAGS)
    accept_usernames = provider in PROVIDERS_SUPPORTING_USERNAMES
    max_text         = cfg["max_text"]
    max_images       = cfg["max_images"] if accept_images else 0
    max_messages     = cfg["max_messages"]
    use_plain        = cfg["use_plain_responses"]
    max_msg_length   = 2000 if use_plain else (4096 - len(STREAMING_INDICATOR))

    # Build conversation history
    messages = []
    user_warnings = set()
    curr_msg = new_msg
    while curr_msg and len(messages) < max_messages:
        node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        async with node.lock:
            if node.text is None:
                cleaned = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()
                good_atts = [
                    att for att in curr_msg.attachments
                    if att.content_type and att.content_type.startswith(("text","image"))
                ]
                atts_data = await asyncio.gather(*[httpx_client.get(att.url) for att in good_atts])
                node.text = "\n".join(
                    ([cleaned] if cleaned else []) +
                    ["\n".join(filter(None,(emb.title,emb.description,emb.footer.text))) for emb in curr_msg.embeds] +
                    [resp.text for att,resp in zip(good_atts, atts_data) if att.content_type.startswith("text")]
                )
                node.images = [
                    dict(type="image_url", image_url=dict(
                        url=f"data:{att.content_type};base64,{b64encode(resp.content).decode()}"
                    ))
                    for att,resp in zip(good_atts, atts_data)
                    if att.content_type.startswith("image")
                ]
                node.role    = "assistant" if curr_msg.author == discord_client.user else "user"
                node.user_id = curr_msg.author.id if node.role=="user" else None
                node.has_bad_attachments = len(curr_msg.attachments) > len(good_atts)
            # assemble payload
            content = ([dict(type="text", text=node.text[:max_text])] if node.text[:max_text] else []) + node.images[:max_images]
            if content:
                msg = dict(content=node.text[:max_text], role=node.role)
                if accept_usernames and node.user_id is not None:
                    msg["name"] = str(node.user_id)
                messages.append(msg)
        curr_msg = node.parent_msg

    # Optional system prompt
    if system_prompt := cfg["system_prompt"]:
        extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            extras.append("Usernames are their Discord IDs as '<@ID>'.")
        full = "\n".join([system_prompt] + extras)
        messages.append(dict(role="system", content=full))

    # Stream response from OpenAI
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    embed = discord.Embed()
    for warn in sorted(user_warnings):
        embed.add_field(name=warn, value="", inline=False)

    kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        extra_body=cfg["extra_api_parameters"]
    )

    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**kwargs):
                if finish_reason is not None:
                    break
                finish_reason = chunk.choices[0].finish_reason
                delta = chunk.choices[0].delta.content or ""
                new_text = (response_contents[-1] if response_contents else "") + delta
                # handle chunk accumulation, editing, embeds, etc.
                response_contents.append(new_text)
                # ... (rest of your streaming/editing logic) ...

    except Exception:
        logging.exception("Error while generating response")

    # Clean up old nodes
    now = dt.now().timestamp()
    if now - last_task_time > EDIT_DELAY_SECONDS:
        # purge stale nodes
        for msg_id, node in list(msg_nodes.items()):
            if (now - getattr(node, "timestamp", now)) > 3600:
                msg_nodes.pop(msg_id, None)

async def main():
    await discord_client.start(cfg["bot_token"])

asyncio.run(main())
