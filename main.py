# =============================================================================
# 免责声明 / Disclaimer
# -----------------------------------------------------------------------------
# 1. 本插件仅供个人学习、技术研究与交流使用，严禁用于任何商业用途或违法违规目的。
#    严禁使用本插件生成、传播任何违反法律法规或侵害他人合法权益的内容。
# 2. 本插件为非官方、非商业的个人学习作品，与 Pollinations、AstrBot 及任何第三方
#    服务提供商均无隶属、合作或授权关系。所调用的 Pollinations 为第三方公共免费
#    服务，其可用性、稳定性、安全性与内容合规性均由该第三方独立负责，与作者无关。
# 3. 本插件按"现状"提供，不作任何明示或默示担保。使用本插件所生成的一切内容，
#    以及由此直接或间接产生的任何后果、损失、纠纷或法律责任，均由使用者本人自行
#    承担，与本插件作者完全无关，作者不承担任何责任亦不做任何赔偿。
# 4. 使用者须自行遵守所在国家/地区的法律法规及第三方服务条款。
# 5. 一旦下载、安装或以任何方式使用本插件，即视为已完整阅读并无条件同意本声明；
#    如不同意其中任何一条，请立即停止使用并彻底删除本插件。
# =============================================================================

import asyncio
import random
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import aiohttp

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register

# 宽高比 -> 像素预设
ASPECT_RATIO_TO_SIZE = {
    "1:1": (1024, 1024),
    "3:4": (768, 1024),
    "4:3": (1024, 768),
    "9:16": (576, 1024),
    "16:9": (1024, 576),
}

VALID_ASPECT_RATIOS = set(ASPECT_RATIO_TO_SIZE.keys())

DEFAULT_BASE_ENDPOINT = "https://image.pollinations.ai/prompt"


@register(
    "astrbot_plugin_imagefree_t2i",
    "menglian001",
    "基于 Pollinations 的免费文生图插件（无需 API Key）",
    "1.0.0",
    "https://github.com/menglian001/astrbot_plugin_imagefree_t2i",
)
class ImageFreePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 图片保存到 data 目录，避免插件更新/重装时被清空
        self._image_dir = Path("data") / "imagefree_t2i"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        # 节流状态：记录上次请求时间戳，并用锁串行化节流判断
        self._last_request_ts: float = 0.0
        self._throttle_lock = asyncio.Lock()

    # ---------- 配置读取辅助 ----------

    def _api_endpoint(self) -> str:
        return str(
            self.config.get("api_endpoint") or DEFAULT_BASE_ENDPOINT
        ).rstrip("/")

    def _default_aspect_ratio(self) -> str:
        ar = str(self.config.get("default_aspect_ratio") or "1:1")
        return ar if ar in VALID_ASPECT_RATIOS else "1:1"

    def _model(self) -> str:
        return str(self.config.get("model") or "flux")

    def _nologo(self) -> bool:
        return bool(self.config.get("nologo", True))

    def _max_retries(self) -> int:
        return max(1, int(self.config.get("max_retries") or 3))

    def _proxy_url(self) -> Optional[str]:
        proxy = self.config.get("proxy_url")
        return proxy if proxy else None

    def _headers(self) -> dict[str, str]:
        ua = str(
            self.config.get("user_agent")
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )
        return {
            "Accept": "image/*,*/*",
            "User-Agent": ua,
        }

    def _request_timeout(self) -> int:
        return int(self.config.get("request_timeout_seconds") or 60)

    def _throttle_interval(self) -> float:
        # 两次生图请求的最小间隔（秒），0 表示不节流
        return max(0.0, float(self.config.get("min_request_interval_seconds") or 0))

    def _backoff_base(self) -> float:
        # 429 退避基数（秒）
        return max(0.5, float(self.config.get("retry_backoff_base_seconds") or 2))

    def _backoff_max(self) -> float:
        # 429 退避封顶（秒）
        return max(1.0, float(self.config.get("retry_backoff_max_seconds") or 30))

    def _cleanup_enabled(self) -> bool:
        return bool(self.config.get("cleanup_enabled", True))

    def _keep_max_files(self) -> int:
        # 缓存目录最多保留图片数，0 表示不按数量清理
        return max(0, int(self.config.get("max_cached_images") or 0))

    def _keep_max_days(self) -> int:
        # 图片保留天数，0 表示不按时间清理
        return max(0, int(self.config.get("cache_retention_days") or 0))

    # ---------- 核心生图逻辑 ----------

    def _build_url(self, prompt: str, width: int, height: int, seed: int) -> str:
        quoted = quote(prompt, safe="")
        params = [
            f"width={width}",
            f"height={height}",
            f"model={quote(self._model(), safe='')}",
            f"seed={seed}",
        ]
        if self._nologo():
            params.append("nologo=true")
        return f"{self._api_endpoint()}/{quoted}?{'&'.join(params)}"

    @staticmethod
    def _is_image_bytes(data: bytes) -> bool:
        if not data or len(data) < 16:
            return False
        # JPEG / PNG / WEBP / GIF 文件头
        return (
            data[:3] == b"\xff\xd8\xff"
            or data[:8] == b"\x89PNG\r\n\x1a\n"
            or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
            or data[:6] in (b"GIF87a", b"GIF89a")
        )

    async def _throttle(self) -> None:
        """请求节流：确保两次生图请求间隔不小于配置的最小间隔。"""
        interval = self._throttle_interval()
        if interval <= 0:
            return
        async with self._throttle_lock:
            now = time.monotonic()
            wait = self._last_request_ts + interval - now
            if wait > 0:
                logger.info(f"[imagefree] 节流等待 {wait:.1f}s 以避免限流")
                await asyncio.sleep(wait)
            # 记录本次请求的“放行时刻”
            self._last_request_ts = time.monotonic()

    def _backoff_delay(self, attempt: int) -> float:
        """429 指数退避：base * 2^(attempt-1)，封顶 max。"""
        delay = self._backoff_base() * (2 ** (attempt - 1))
        return min(delay, self._backoff_max())

    async def _generate_image(
        self, prompt: str, aspect_ratio: Optional[str] = None
    ) -> bytes:
        """向 Pollinations 提交生图请求并返回图片字节，失败时抛出异常。"""
        aspect_ratio = aspect_ratio or self._default_aspect_ratio()
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            aspect_ratio = self._default_aspect_ratio()
        width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]

        # 进入前先节流，从源头降低触发 429 的概率
        await self._throttle()

        timeout = aiohttp.ClientTimeout(total=self._request_timeout())
        last_error: Optional[str] = None

        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(1, self._max_retries() + 1):
                seed = random.randint(1, 2_147_483_647)
                url = self._build_url(prompt, width, height, seed)
                try:
                    async with session.get(
                        url,
                        headers=self._headers(),
                        proxy=self._proxy_url(),
                    ) as resp:
                        if resp.status == 429:
                            # 限流：指数退避后重试
                            last_error = "HTTP 429 Too Many Requests"
                            delay = self._backoff_delay(attempt)
                            logger.warning(
                                f"[imagefree] 第 {attempt} 次遇到 429 限流，"
                                f"退避 {delay:.1f}s 后重试"
                            )
                            await asyncio.sleep(delay)
                            continue
                        if resp.status != 200:
                            last_error = f"HTTP {resp.status}"
                            logger.warning(
                                f"[imagefree] 第 {attempt} 次请求返回 {resp.status}"
                            )
                            await asyncio.sleep(1)
                            continue
                        data = await resp.read()
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = str(e)
                    logger.warning(f"[imagefree] 第 {attempt} 次请求异常: {e}")
                    await asyncio.sleep(1)
                    continue

                if self._is_image_bytes(data):
                    return data

                last_error = f"返回内容非图片或为空（{len(data)} 字节）"
                logger.warning(f"[imagefree] 第 {attempt} 次: {last_error}")
                await asyncio.sleep(1)

        raise RuntimeError(f"生图失败，已重试 {self._max_retries()} 次：{last_error}")

    def _save_image(self, image_bytes: bytes) -> Path:
        # 依据文件头判断后缀，默认 png
        suffix = ".png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            suffix = ".jpg"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            suffix = ".webp"
        elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            suffix = ".gif"
        target = self._image_dir / f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
        target.write_bytes(image_bytes)
        self._cleanup_images()
        return target

    _IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif")

    def _cleanup_images(self) -> None:
        """清理缓存目录中的旧图：先删超期文件，再裁剪至数量上限。

        仅操作本插件目录内的图片文件；任何异常仅记日志，不影响出图主流程。
        """
        if not self._cleanup_enabled():
            return
        max_files = self._keep_max_files()
        max_days = self._keep_max_days()
        if max_files <= 0 and max_days <= 0:
            return
        try:
            files = [
                p
                for p in self._image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._IMAGE_SUFFIXES
            ]
        except OSError as e:
            logger.warning(f"[imagefree] 清理时遍历目录失败: {e}")
            return

        # 1) 按保留天数删除超期文件
        if max_days > 0:
            cutoff = time.time() - max_days * 86400
            remaining = []
            for p in files:
                try:
                    if p.stat().st_mtime < cutoff:
                        p.unlink()
                        logger.info(f"[imagefree] 清理超期图片: {p.name}")
                    else:
                        remaining.append(p)
                except OSError as e:
                    logger.warning(f"[imagefree] 删除超期文件失败 {p.name}: {e}")
                    remaining.append(p)
            files = remaining

        # 2) 按数量上限裁剪（保留最新的 max_files 个）
        if max_files > 0 and len(files) > max_files:
            try:
                files.sort(key=lambda p: p.stat().st_mtime)
            except OSError as e:
                logger.warning(f"[imagefree] 清理排序失败: {e}")
                return
            for p in files[: len(files) - max_files]:
                try:
                    p.unlink()
                    logger.info(f"[imagefree] 清理超量图片: {p.name}")
                except OSError as e:
                    logger.warning(f"[imagefree] 删除超量文件失败 {p.name}: {e}")

    def _count_cached_images(self) -> int:
        """统计缓存目录内的图片文件数量，出错时返回 0。"""
        try:
            return sum(
                1
                for p in self._image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._IMAGE_SUFFIXES
            )
        except OSError:
            return 0

    # ---------- 指令 ----------

    @filter.command("imagefree_draw", alias={"生图", "画图"})
    async def imagefree_draw(self, event: AstrMessageEvent, prompt: str):
        """使用 Pollinations 免费生成图片。用法: /imagefree_draw <提示词>"""
        prompt = (prompt or "").strip()
        if not prompt:
            yield event.plain_result("请提供图片描述，例如: /imagefree_draw 夕阳下的雪山湖泊")
            return

        yield event.plain_result(f"正在生成图片: {prompt} ...")
        try:
            image_bytes = await self._generate_image(prompt)
            path = self._save_image(image_bytes)
            yield event.image_result(str(path))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[imagefree] 生图失败: {e}")
            yield event.plain_result(f"生图失败: {e}")

    @filter.command("imagefree_status")
    async def imagefree_status(self, event: AstrMessageEvent):
        """查看当前 imagefree 插件的接口与生图配置。"""
        throttle = self._throttle_interval()
        cleanup = (
            f"保留最近 {self._keep_max_files() or '∞'} 张 / "
            f"{self._keep_max_days() or '∞'} 天"
            if self._cleanup_enabled()
            else "已关闭"
        )
        info = (
            "ImageFree 文生图插件状态\n"
            f"- 生图接口: {self._api_endpoint()}\n"
            f"- 模型: {self._model()}\n"
            f"- 默认宽高比: {self._default_aspect_ratio()}\n"
            f"- 去水印(nologo): {'开' if self._nologo() else '关'}\n"
            f"- 最大重试: {self._max_retries()} 次\n"
            f"- 请求节流: {f'{throttle:.0f}s' if throttle > 0 else '关闭'}\n"
            f"- 429 退避: 基数 {self._backoff_base():.0f}s / 封顶 {self._backoff_max():.0f}s\n"
            f"- 自动清理: {cleanup}\n"
            f"- 当前缓存图片: {self._count_cached_images()} 张\n"
            f"- 代理: {self._proxy_url() or '未配置'}\n"
            f"- 单次超时: {self._request_timeout()}s\n"
            f"- 缓存目录: {self._image_dir}"
        )
        yield event.plain_result(info)

    # ---------- 供 LLM 调用的函数工具 ----------

    @filter.llm_tool(name="imagefree_generate_image")
    async def imagefree_generate_image(
        self, event: AstrMessageEvent, prompt: str, aspect_ratio: str = "1:1"
    ):
        """根据文字提示词生成一张图片并发送到当前聊天。

        Args:
            prompt(string): 用于生成图片的文字描述（英文效果通常更好）。
            aspect_ratio(string): 图片宽高比，可选 1:1 / 3:4 / 4:3 / 9:16 / 16:9，默认 1:1。
        """
        if not self.config.get("enable_llm_tool", True):
            return
        try:
            image_bytes = await self._generate_image(prompt, aspect_ratio)
            path = self._save_image(image_bytes)
            yield event.image_result(str(path))
        except Exception as e:  # noqa: BLE001
            logger.error(f"[imagefree] LLM 工具生图失败: {e}")
            yield event.plain_result(f"生图失败: {e}")

    async def terminate(self):
        """插件卸载/停用时调用。"""
        logger.info("[imagefree] 插件已卸载")
