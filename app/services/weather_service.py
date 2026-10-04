import httpx
from typing import Dict, Optional
from app.core.config import settings

# WMO 天气代码 → 中文描述（Open-Meteo 兜底数据源用）
WMO_CODE_ZH = {
    0: "晴", 1: "基本晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹",
}


class WeatherService:
    """
    天气查询服务
    优先使用和风天气 API；未配置 key 时走 Open-Meteo 免费数据源兜底（无需 key，国内可达）
    """

    BASE_URL = "https://devapi.qweather.com/v7"

    def __init__(self):
        self.api_key = settings.WEATHER_API_KEY

    async def get_weather(self, location: str) -> Dict:
        """
        查询实时天气
        :param location: 城市名称或 LocationID（如 "北京" 或 "101010100"）
        :return: 天气信息
        """
        # 未配置和风天气 key 时，走 Open-Meteo 免费数据源兜底
        if not self.api_key:
            return await self._get_open_meteo(location)

        # 如果是中文城市名，先查 LocationID
        location_id = location
        if not location.isdigit():
            location_id = await self._get_location_id(location)
            if not location_id:
                return {"error": f"未找到城市: {location}"}

        url = f"{self.BASE_URL}/weather/now"
        params = {"location": location_id, "key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            # 和风挂了，降级到 Open-Meteo
            return await self._get_open_meteo(location)

        if data.get("code") != "200":
            return {"error": f"天气查询失败: {data.get('code')}"}

        now = data["now"]
        return {
            "source": "qweather",
            "city": location,
            "temp": now["temp"],
            "text": now["text"],
            "windDir": now["windDir"],
            "windScale": now["windScale"],
            "humidity": now["humidity"],
            "obsTime": now["obsTime"],
        }

    async def _get_location_id(self, city_name: str) -> Optional[str]:
        """根据城市名查询和风 LocationID"""
        url = f"{self.BASE_URL}/city/lookup"
        params = {"location": city_name, "key": self.api_key}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return None

        if data.get("code") == "200" and data.get("location"):
            return data["location"][0]["id"]
        return None

    async def _get_open_meteo(self, location: str) -> Dict:
        """Open-Meteo 免费天气（无需 API key，国内可达）：先地理编码再查实况"""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                # 1. 城市名 → 经纬度
                geo = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "language": "zh", "format": "json"},
                )
                geo.raise_for_status()
                geo_data = geo.json()
                if not geo_data.get("results"):
                    return {"error": f"未找到城市: {location}"}
                place = geo_data["results"][0]

                # 2. 经纬度 → 实况天气
                weather = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
                        "timezone": "auto",
                    },
                )
                weather.raise_for_status()
                cur = weather.json()["current"]
        except Exception as e:
            return {"error": f"天气查询失败(Open-Meteo): {e}"}

        code = cur.get("weather_code")
        region = " ".join(x for x in [place.get("country", ""), place.get("admin1", ""), place.get("name", "")] if x)
        return {
            "source": "open-meteo",
            "city": region or location,
            "temp": cur.get("temperature_2m"),
            "text": WMO_CODE_ZH.get(code, f"weather_code {code}"),
            "windDir": cur.get("wind_direction_10m", ""),
            "windScale": cur.get("wind_speed_10m", ""),
            "humidity": cur.get("relative_humidity_2m", ""),
        }


weather_service = WeatherService()