import { app } from "../../../scripts/app.js";

// Базовая схема на случай отсутствия совпадений
const DEFAULT_SCHEME = {
    color: "#2e2e36",
    bgcolor: "#41414a",
    titleText: "#e5e9f0"
};

// Тематические палитры
const SCHEMES = {
    jpg: {            
        color: "#a7a7a7",
        bgcolor: "#c0c0c0",
        titleText: "#f4f3ee"
    }
};

// Перекрытия по категориям
const CATEGORY_SCHEMES = {
    "😎 SnJake/JPG & Noise Remover": SCHEMES.jpg,
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
