import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

import discord
import httpx
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
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

# ——— Config loader ———
def get_config(filename="config.yaml"):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

cfg = get_config()

if client_id := cfg.get("client_id"):
    logging.info(
        f"\n\nBOT INVITE URL:\n"
        f"https://discord.com/api/oauth2/authorize?"
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

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@discord_client.event
async def on_message(new_msg):
    global msg_nodes

    # Reload config for hot-reload
    cfg = get_config()
    use_n8n     = cfg.get("use_n8n", False)
    n8n_webhook = cfg.get("n8n_webhook_url", "")

    # Ignore bots; require mention in guilds or allow DMs
    is_dm = new_msg.channel.type == discord.ChannelType.private
    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    # Permissions check
    role_ids = {r.id for r in getattr(new_msg.author, "roles", [])}
    channel_ids = set(filter(None, (
        new_msg.channel.id,
        getattr(new_msg.channel, "parent_id", None),
        getattr(new_msg.channel, "category_id", None),
    )))
    allow_dms = cfg["allow_dms"]
    perms = cfg["permissions"]
    (allowed_users, blocked_users), (allowed_roles, blocked_roles), (allowed_chs, blocked_chs) = (
        (perm["allowed_ids"], perm["blocked_ids"])
        for perm in (perms["users"], perms["roles"], perms["channels"])
    )
    good_user = (not allowed_users or new_msg.author.id in allowed_users or bool(role_ids & set(allowed_roles)))
    bad_user  = new_msg.author.id in blocked_users or bool(role_ids & set(blocked_roles))
    good_ch   = (allow_dms if is_dm else not allowed_chs or bool(channel_ids & set(allowed_chs)))
    bad_ch    = bool(channel_ids & set(blocked_chs))
    if bad_user or bad_ch or not good_user or not good_ch:
        return

    # ─── n8n branch ───
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
            try:
                data = resp.json()
            except ValueError:
                logging.error("n8n returned invalid JSON: %r", resp.text)
                return

            # handle array vs object
            if isinstance(data, list) and data:
                data = data[0]

            reply = data.get("reply")
            if reply:
                await new_msg.reply(reply)
        else:
            logging.error(f"n8n webhook failed: {resp.status_code} {resp.text}")
        return

    # ─── Fallback: original OpenAI streaming logic ───
    from openai import AsyncOpenAI  # delayed import
    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key  = cfg["providers"][provider].get("api_key", "")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(tag in model.lower() for tag in VISION_MODEL_TAGS)
    max_text      = cfg["max_text"]
    max_images    = cfg["max_images"] if accept_images else 0
    max_messages  = cfg["max_messages"]
    use_plain     = cfg["use_plain_responses"]

    # Build conversation history (simplified for brevity)
    messages = []
    curr = new_msg
    while curr and len(messages) < max_messages:
        node = msg_nodes.setdefault(curr.id, MsgNode())
        async with node.lock:
            if node.text is None:
                cleaned = curr.content.removeprefix(discord_client.user.mention).strip()
                node.text = cleaned or ""
                node.role = "assistant" if curr.author == discord_client.user else "user"
                node.user_id = curr.author.id if node.role == "user" else None
            if node.text:
                messages.append({"role": node.role, "content": node.text[:max_text]})
        curr = node.parent_msg

    # Optional system prompt
    if sp := cfg.get("system_prompt"):
        messages.append({"role": "system", "content": sp})

    # Stream from OpenAI
    kwargs = {
        "model": model,
        "messages": messages[::-1],
        "stream": True,
        "extra_body": cfg.get("extra_api_parameters", {}),
    }
    try:
        async with new_msg.channel.typing():
            async for chunk in await openai_client.chat.completions.create(**kwargs):
                # ... existing chunk handling/editing logic ...
                pass
    except Exception:
        logging.exception("Error in OpenAI streaming")

async def main():
    await discord_client.start(cfg["bot_token"])

asyncio.run(main())
