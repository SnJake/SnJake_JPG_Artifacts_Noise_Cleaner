import { app } from "../../../scripts/app.js";

// Базовая схема на случай отсутствия совпадений
const DEFAULT_SCHEME = {
    color: "#2e2e36",
    bgcolor: "#41414a",
    titleText: "#e5e9f0"
};

// Тематические палитры
const SCHEMES = {
    vlm: {            // Фиолетово-индиговая — VLM / мультимодальность
        color: "#3b2a6f",
        bgcolor: "#201833",
        titleText: "#efeaff"
    },
    utils: {          // Холодный стальной — утилиты/служебные
        color: "#2f4858",
        bgcolor: "#17232b",
        titleText: "#e6f8ff"
    },
    effects: {        // Неоновые эффекты — постпроцесс
        color: "#7a1fa2",
        bgcolor: "#2b1035",
        titleText: "#fdeaff"
    },
    adjustment: {     // Янтарно-медный — коррекция
        color: "#7c3f14",
        bgcolor: "#24160e",
        titleText: "#ffe3c2"
    },
    masks: {          // Тёмный бирюзовый — маски/сегментация
        color: "#0d4d4d",
        bgcolor: "#062a2a",
        titleText: "#d8ffff"
    },
    anynode: {        // Графит — обёртки/AnyNode
        color: "#3c3c3c",
        bgcolor: "#1f1f1f",
        titleText: "#f0f0f0"
    },
    pixelart: {       // Ретро-зелёный — пиксель-арт
        color: "#1f6f3e",
        bgcolor: "#0f2317",
        titleText: "#e7ffef"
    },
    xyplot: {         // Холодный синий — графики/плоты
        color: "#12476b",
        bgcolor: "#0a273a",
        titleText: "#dbf2ff"
    },
    lora: {           // Фуксия — обучение/адаптация
        color: "#8b1e5c",
        bgcolor: "#2b0d20",
        titleText: "#ffd7eb"
    },
    detailer: {       // Изумруд — детект/детализация
        color: "#2b6f5b",
        bgcolor: "#102b24",
        titleText: "#e0fff7"
    },
    yolo: {           // Чёрно-золотой — детекция/YOLO
        color: "#6f5500",
        bgcolor: "#1a1503",
        titleText: "#ffeaa6"
    }
};

// Перекрытия по категориям
const CATEGORY_SCHEMES = {
    "😎 SnJake/VLM": SCHEMES.vlm,
    "😎 SnJake/Utils": SCHEMES.utils,
    "😎 SnJake/Effects": SCHEMES.effects,
    "😎 SnJake/Adjustment": SCHEMES.adjustment,
    "😎 SnJake/Masks": SCHEMES.masks,
    "😎 SnJake/AnyNode": SCHEMES.anynode,
    "😎 SnJake/PixelArt": SCHEMES.pixelart,
    "😎 SnJake/XY Plot": SCHEMES.xyplot,
    "😎 SnJake/LoRA": SCHEMES.lora,
    "😎 SnJake/Detailer": SCHEMES.detailer,
    "😎 SnJake/YOLO": SCHEMES.yolo
};

function pickScheme(node) {
    const cat = (node?.constructor?.category || node?.category || "").toString();
    for (const key of Object.keys(CATEGORY_SCHEMES)) {
        if (cat.startsWith(key)) return CATEGORY_SCHEMES[key];
    }
    return DEFAULT_SCHEME;
}

function applyColors(node) {
    const s = pickScheme(node);
    node.color = s.color;
    node.bgcolor = s.bgcolor;
    try {
        if (node?.constructor) {
            node.constructor.title_text_color = s.titleText;
        }
    } catch {}
}

app.registerExtension({
    name: "SnJake.AutoColors",
    async nodeCreated(node) {
        const title = (node && (node.title || "")).toString();
        if (!title.startsWith("😎")) return;
        applyColors(node);
    }
});
