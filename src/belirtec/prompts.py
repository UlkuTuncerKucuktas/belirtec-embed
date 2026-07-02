from __future__ import annotations

import random

from belirtec.config import Axes


class AxisSampler:
    def __init__(self, axes: Axes, seed: int = 42):
        self.axes = axes
        self.rng = random.Random(seed)
        self._difficulty_pool = [
            level for level, weight in axes.difficulty.items() for _ in range(max(1, weight))
        ]

    def sample(self) -> dict[str, str]:
        return {
            "persona": self.rng.choice(self.axes.persona),
            "intent": self.rng.choice(self.axes.intent),
            "difficulty": self.rng.choice(self._difficulty_pool),
        }


_DIFFICULTY_HINT = {
    "doğrudan": "Metindeki terimleri kullanabilirsin; soruyu doğrudan sor.",
    "dolaylı": "Metindeki anahtar terimleri kullanma; aynı bilgiyi farklı sözcüklerle sor.",
    "çıkarımsal": "Metnin doğrudan söylemediği ama içinden çıkarılabilecek bir sonucu veya imayı sor.",
}


def grounded_query_prompt(passage: str, axis: dict[str, str], domain: str) -> str:
    persona, intent, difficulty = axis["persona"], axis["intent"], axis["difficulty"]
    diff_hint = _DIFFICULTY_HINT.get(difficulty, "")
    kind = "hukuk metni (mahkeme kararı/mevzuat)" if domain == "hukuki" else "Türkçe metin"
    return (
        f"Aşağıda gerçek bir {kind} var. Bu metnin CEVAPLADIĞI tek bir gerçekçi Türkçe SORU yaz.\n"
        f"Soruyu şu kişi soruyor: {persona}. Sorunun amacı: {intent}.\n"
        f"{diff_hint}\n"
        f"Kurallar: soru doğal ve kısa-orta uzunlukta olsun; metindeki cümleleri aynen kopyalama; "
        f"ezber bilgi (karar/madde numarası) gerektirme; akıcı ve doğru Türkçe kullan.\n"
        f'Yalnızca JSON döndür: {{"query": "<soru>"}}\n\n'
        f"METİN:\n{passage}"
    )


def sts_topics_prompt(n: int, angle: str) -> str:
    return (
        f"{angle} alanında, anlamsal benzerlik cümle çiftleri için {n} farklı kısa Türkçe "
        f"konu başlığı üret. Akıcı, doğru Türkçe. Yalnızca JSON dizisi (string listesi) döndür."
    )


def sts_samples_prompt(topic: str, n: int) -> str:
    return (
        f"Konu: {topic}\n{n} çeşitli örnek üret. Her örnek için:\n"
        f'  "text_a": doğal bir Türkçe cümle\n'
        f'  "text_b": text_a ile AYNI anlama gelen, farklı sözcüklerle yazılmış cümle\n'
        f'  "hard_negatives": aynı konuda ama FARKLI anlam taşıyan 1-3 cümle\n'
        f"text_a ile text_b birebir aynı OLMASIN. Yalnızca JSON listesi: "
        f'[{{"text_a":"...","text_b":"...","hard_negatives":["..."]}}]'
    )


def cls_tasks_prompt(n: int, domain: str) -> str:
    return (
        f"Türkçe metin sınıflandırma görevleri tasarlıyorsun. Bu partide odak: {domain}. "
        f"{n} FARKLI görev üret. Her görev için: \"task\" (neyin sınıflandırılacağını anlatan "
        f"doğal Türkçe talimat) ve \"labels\" (2-5 birbirini dışlayan Türkçe etiket). "
        f"Etiketler gerçek Türkçe sözcükler olsun; yabancı sözcük yok. "
        f'Yalnızca JSON dizisi. Örnek: '
        f'{{"task":"Müşteri mesajının aciliyetini sınıflandır","labels":["acil","normal","düşük"]}}'
    )


def cls_samples_prompt(task: str, labels: list[str], target: str, n: int) -> str:
    others = ", ".join(f'"{l}"' for l in labels if l != target) or "(yok)"
    return (
        f"Sınıflandırma görevi: {task}\nEtiketler: {labels}\nHedef etiket: \"{target}\"\n"
        f"Doğru etiketi \"{target}\" olan {n} kısa doğal çeşitli Türkçe metin yaz.\n"
        f"Metinde etiket adını asla söyleme; konu/uzunluk çeşitli olsun.\n"
        f"Her metin için karışabilecek 2-4 yanlış etiket ver (şu kümeden: [{others}]).\n"
        f'Yalnızca JSON listesi: [{{"text":"...","misleading_labels":["...","..."]}}]'
    )


_FOREIGN_LABELS = {"neutral", "positive", "negative", "neutro", "yes", "no", "true", "false"}


def is_foreign_label(label: str) -> bool:
    return label.strip().lower() in _FOREIGN_LABELS


STS_ANGLES = [
    "günlük yaşam ve sohbet", "haber ve güncel olaylar", "ürün ve hizmet yorumları",
    "bilim ve teknoloji", "duygular, görüşler ve tutumlar", "iş ve resmî yazışma",
]
CLS_DOMAINS = [
    "duygu/görüş analizi, müşteri memnuniyeti, şikâyet türleri",
    "konu/kategori sınıflandırması, alan tespiti, metin türü",
    "niyet tespiti, amaç, soru türü, talep türü",
    "üslup/resmiyet, ton, hedef kitle, aciliyet",
    "duygu durumu (öfke, sevinç, üzüntü, korku), ruh hâli, tutum",
]
STS_INSTRUCTION = "Anlamca benzer metni getir."
