import { app } from "../../../scripts/app.js";

// Decorative slider ONLY for the JPG & Noise Remover node
app.registerExtension({
    name: "SnJake.DecorSlider.JpgOnly",
    async nodeCreated(node) {
        // ---- strict target check ----
        const cls = String(node?.comfyClass || node?.type || node?.constructor?.name || "");
        const title = String(node?.title || "");
        const category = String(node?.category || "");

        // Match by comfyClass / constructor first
        const isClassMatch =
            cls === "SnJakeArtifactsRemover" ||
            node?.constructor?.name === "SnJakeArtifactsRemover";

        // Fallback: match by human-facing labels
        const re = /jpg\s*&?\s*noise\s*remover/i;
        const isLabelMatch = re.test(title) || re.test(category);

        const TARGET = isClassMatch || isLabelMatch;
        if (!TARGET) return;

        // avoid interfering with anything else
        const FLAG = "__sjake_anim_init_jpgnr";
        if (node[FLAG]) return;
        node[FLAG] = true;

        // ---- Waves toggle (default: false) ----
        node.__sjake_waves_enabled = false;
        if (typeof node.addWidget === "function") {
            // Don't add twice
            const hasWidget = (node.widgets || []).some(w => w?.name === "Decor Waves");
            if (!hasWidget) {
                const w = node.addWidget(
                    "toggle",
                    "Decor Waves",
                    false,
                    (v) => { node.__sjake_waves_enabled = !!v; }
                );
                // ensure internal flag mirrors widget value
                node.__sjake_waves_enabled = !!(w?.value);
            }
        }

        // keep originals
        const prevOnDrawForeground = node.onDrawForeground?.bind(node);
        const prevOnRemoved = node.onRemoved?.bind(node);

        node.__sjake_phase_offset = Math.random() * Math.PI * 2;
        node.__sjake_waves = [];
        node.__sjake_last_time = performance.now() / 1000;
        node.__sjake_next_pulse = node.__sjake_last_time + 2 + Math.random() * 4;

        node.onDrawForeground = function (ctx) {
            if (prevOnDrawForeground) prevOnDrawForeground(ctx);

            const w = this.size?.[0] ?? 0;
            const h = this.size?.[1] ?? 0;
            if (!w || !h) return;
            if (this.flags && (this.flags.collapsed || this.flags.minimized)) return;

            const pad = 8;
            const trackH = 6;
            const radius = 3;
            const x = pad;
            const y = h + 6; // draw below node
            const trackW = Math.max(0, w - pad * 2);

            const now = performance.now() / 1000;
            const t = now * 1.2 + (this.__sjake_phase_offset || 0);
            const pingpong = 0.5 * (1 + Math.sin(t));
            const thumbW = Math.max(16, Math.min(28, trackW * 0.2));
            const thumbX = x + pingpong * (trackW - thumbW);

            const theme = computeDecorColors(this);

            // track
            ctx.save();
            ctx.globalAlpha = 0.35;
            roundRect(ctx, x, y, trackW, trackH, radius);
            ctx.fillStyle = theme.track;
            ctx.fill();
            ctx.restore();

            // waves (controlled by toggle)
            if (this.__sjake_waves_enabled) {
                const dt = Math.min(0.1, Math.max(0, now - (this.__sjake_last_time || now)));
                this.__sjake_last_time = now;
                if (now >= (this.__sjake_next_pulse || now)) {
                    spawnWaves(this, thumbX + thumbW * 0.5, x, x + trackW, theme);
                    this.__sjake_next_pulse = now + 3 + Math.random() * 4;
                }
                drawWaves(ctx, this, x, y, trackW, trackH, theme, dt);
            } else {
                // keep memory clean when disabled
                this.__sjake_waves = [];
                this.__sjake_next_pulse = now + 3 + Math.random() * 4;
                this.__sjake_last_time = now;
            }

            // thumb
            const pulse = 1 + 0.06 * Math.sin(now * 0.9 + (this.__sjake_phase_offset || 0));
            const thumbH = (trackH + 1) * pulse;
            const thumbY = y - 0.5 - (thumbH - (trackH + 1)) * 0.5;

            const grad = ctx.createLinearGradient(thumbX, y, thumbX + thumbW, y);
            grad.addColorStop(0.0, theme.thumbLight);
            grad.addColorStop(1.0, theme.thumbDark);

            ctx.save();
            ctx.shadowColor = "rgba(0,0,0,0.25)";
            ctx.shadowBlur = 2;
            ctx.shadowOffsetY = 1;
            roundRect(ctx, thumbX, thumbY, thumbW, thumbH, radius);
            ctx.fillStyle = grad;
            ctx.fill();
            ctx.restore();

            ctx.save();
            ctx.globalAlpha = 0.35;
            ctx.beginPath();
            ctx.moveTo(thumbX + 1, thumbY + 1);
            ctx.lineTo(thumbX + thumbW - 1, thumbY + 1);
            ctx.strokeStyle = theme.thumbHighlight;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        };

        // continuous redraw while node exists
        const ensureRedraw = () => {
            const canvas = app?.canvas;
            if (!canvas) return;
            if (typeof canvas.setDirty === "function") canvas.setDirty(true, true);
            else if (typeof canvas.draw === "function") canvas.draw(true, true);
        };
        const animate = () => {
            if (!node.graph) return;
            ensureRedraw();
            node.__sjake_anim_frame = requestAnimationFrame(animate);
        };
        node.__sjake_anim_frame = requestAnimationFrame(animate);

        node.onRemoved = function () {
            if (prevOnRemoved) prevOnRemoved();
            if (this.__sjake_anim_frame) cancelAnimationFrame(this.__sjake_anim_frame);
            this.__sjake_anim_frame = null;
        };
    }
});

// helpers
function roundRect(ctx, x, y, w, h, r) {
    const rr = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
}

function computeDecorColors(node) {
    const bg = safeColor(node?.bgcolor, "#41414a");
    const base = safeColor(node?.color, "#2e2e36");
    const title = safeColor(node?.constructor?.title_text_color, "#e5e9f0");
    const track = mixHex(bg, base, 0.25, 0.5);
    const thumbLight = lightenHex(base, 0.28);
    const thumbDark = darkenHex(base, 0.12);
    const wave = setAlphaHex(title, 0.22);
    const waveStrong = setAlphaHex(title, 0.35);
    return { track, thumbLight, thumbDark, thumbHighlight: setAlphaHex("#ffffff", 0.35), wave, waveStrong };
}
function safeColor(v, fb) { return (typeof v === "string" && /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(v)) ? v : fb; }
function hexToRgb(hex) { hex = hex.replace(/^#/, ""); if (hex.length === 3) hex = hex.split("").map(c => c + c).join(""); const int = parseInt(hex, 16); return { r: (int >> 16) & 255, g: (int >> 8) & 255, b: int & 255 }; }
function rgbToHex(r, g, b) { const to = (n) => Math.max(0, Math.min(255, n | 0)).toString(16).padStart(2, "0"); return `#${to(r)}${to(g)}${to(b)}`; }
function mixHex(a, b, t, alpha = 1) { const A = hexToRgb(a), B = hexToRgb(b); const r = A.r + (B.r - A.r) * t; const g = A.g + (B.g - A.g) * t; const bl = A.b + (B.b - A.b) * t; const base = rgbToHex(r, g, bl); return setAlphaHex(base, alpha); }
function lightenHex(hex, k) { const c = hexToRgb(hex); return rgbToHex(c.r + (255 - c.r) * k, c.g + (255 - c.g) * k, c.b + (255 - c.b) * k); }
function darkenHex(hex, k) { const c = hexToRgb(hex); return rgbToHex(c.r * (1 - k), c.g * (1 - k), c.b * (1 - k)); }
function setAlphaHex(hex, a) { const { r, g, b } = hexToRgb(hex); const aa = Math.max(0, Math.min(1, a)); return `rgba(${r},${g},${b},${aa})`; }

function spawnWaves(node, centerX, minX, maxX, theme) {
    const speed = 80 + Math.random() * 60;
    const width = 8;
    const life = 2.2;
    const now = performance.now() / 1000;
    const base = { start: now, x: centerX, speed, width, life, color: theme.wave, strong: theme.waveStrong };
    node.__sjake_waves.push({ ...base, dir: -1, minX, maxX });
    node.__sjake_waves.push({ ...base, dir: +1, minX, maxX });
    if (node.__sjake_waves.length > 12) node.__sjake_waves.splice(0, node.__sjake_waves.length - 12);
}

function drawWaves(ctx, node, x, y, w, h, theme, dt) {
    const now = performance.now() / 1000;
    const waves = node.__sjake_waves || [];
    for (let i = waves.length - 1; i >= 0; i--) {
        const wave = waves[i];
        const t = now - wave.start;
        if (t > wave.life) { waves.splice(i, 1); continue; }
        const eased = t / wave.life;
        const alphaScale = 1 - eased;
        const curX = wave.x + wave.dir * wave.speed * t;
        if (curX < wave.minX - 12 || curX > wave.maxX + 12) { waves.splice(i, 1); continue; }

        const stripeW = wave.width * (1 + 0.2 * eased);
        const gx0 = curX - stripeW * 0.5;
        const gx1 = curX + stripeW * 0.5;
        const grad = ctx.createLinearGradient(gx0, 0, gx1, 0);
        grad.addColorStop(0.0, "rgba(0,0,0,0)");
        grad.addColorStop(0.5, blendAlpha(wave.strong, alphaScale));
        grad.addColorStop(1.0, "rgba(0,0,0,0)");

        ctx.save();
        ctx.beginPath();
        roundRect(ctx, x, y, w, h, 3);
        ctx.clip();
        ctx.fillStyle = grad;
        ctx.fillRect(gx0, y - 2, stripeW, h + 4);
        ctx.restore();
    }
}

function blendAlpha(rgba, scale) {
    const m = rgba.match(/^rgba\((\d+),(\d+),(\d+),(\d*\.?\d+)\)$/);
    if (!m) return rgba;
    const r = +m[1], g = +m[2], b = +m[3], a = +m[4];
    const na = Math.max(0, Math.min(1, a * scale));
    return `rgba(${r},${g},${b},${na})`;
}
