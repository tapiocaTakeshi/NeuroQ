#!/usr/bin/env python3
"""
NeuroQ 日次時間制限モジュール
==============================
1日あたりの累計使用時間を制限する機能を提供

デフォルト: 1日（86400秒）
- 各リクエストの処理時間を累計
- 日付が変わるとリセット
- 制限超過時はエラーを返す
"""

import time
import threading
from datetime import datetime, date


class DailyTimeLimiter:
    """
    日次時間制限管理クラス

    1日あたりの累計処理時間を追跡し、制限を超えた場合にリクエストを拒否する。
    日付が変わると自動的にリセットされる。
    """

    def __init__(self, daily_limit_seconds: int = 86400):
        """
        Args:
            daily_limit_seconds: 1日あたりの制限時間（秒）。デフォルト86400秒（24時間）
        """
        self.daily_limit_seconds = daily_limit_seconds
        self._used_seconds = 0.0
        self._current_date = date.today()
        self._lock = threading.Lock()
        self._active_requests = 0

    def _reset_if_new_day(self):
        """日付が変わっていたらカウンターをリセット"""
        today = date.today()
        if today != self._current_date:
            self._used_seconds = 0.0
            self._current_date = today

    def check_available(self) -> dict:
        """
        残り時間を確認

        Returns:
            dict: {
                "allowed": bool,
                "used_seconds": float,
                "remaining_seconds": float,
                "daily_limit_seconds": int,
                "date": str,
                "active_requests": int
            }
        """
        with self._lock:
            self._reset_if_new_day()
            remaining = max(0.0, self.daily_limit_seconds - self._used_seconds)
            return {
                "allowed": remaining > 0,
                "used_seconds": round(self._used_seconds, 2),
                "remaining_seconds": round(remaining, 2),
                "daily_limit_seconds": self.daily_limit_seconds,
                "date": str(self._current_date),
                "active_requests": self._active_requests,
            }

    def start_request(self) -> bool:
        """
        リクエスト開始を記録

        Returns:
            bool: リクエストが許可されたかどうか
        """
        with self._lock:
            self._reset_if_new_day()
            if self._used_seconds >= self.daily_limit_seconds:
                return False
            self._active_requests += 1
            return True

    def end_request(self, elapsed_seconds: float):
        """
        リクエスト終了を記録し、経過時間を加算

        Args:
            elapsed_seconds: リクエストの処理にかかった秒数
        """
        with self._lock:
            self._used_seconds += elapsed_seconds
            self._active_requests = max(0, self._active_requests - 1)

    def set_limit(self, daily_limit_seconds: int):
        """
        日次制限時間を変更

        Args:
            daily_limit_seconds: 新しい制限時間（秒）
        """
        with self._lock:
            self.daily_limit_seconds = daily_limit_seconds

    def get_status(self) -> dict:
        """
        詳細なステータス情報を取得

        Returns:
            dict: ステータス情報
        """
        info = self.check_available()
        hours_used = info["used_seconds"] / 3600
        hours_remaining = info["remaining_seconds"] / 3600
        hours_limit = info["daily_limit_seconds"] / 3600
        info["used_hours"] = round(hours_used, 2)
        info["remaining_hours"] = round(hours_remaining, 2)
        info["daily_limit_hours"] = round(hours_limit, 2)
        return info
