import { app } from "../../../scripts/app.js";

// Universal decorative silver slider for all SnJake nodes (titles prefixed with "😎")
app.registerExtension({
    name: "SnJake.DecorSlider",
    async nodeCreated(node) {
        // Only target SnJake nodes that use the 😎 prefix in their title
        const title = (node && (node.title || "")).toString();
        const isSnJake = title.startsWith("😎");
        if (!isSnJake) return;

        // Avoid double-initialization if a node-specific script already added it
        if (node.__sjake_anim_init) return;

        node.__sjake_anim_init = true;

        // Keep original hooks
        const prevOnDrawForeground = node.onDrawForeground?.bind(node);
        const prevOnRemoved = node.onRemoved?.bind(node);

        node.__sjake_phase_offset = Math.random() * Math.PI * 2;

        node.onDrawForeground = function (ctx) {
            // Call original foreground drawing first
            if (prevOnDrawForeground) prevOnDrawForeground(ctx);

            const w = this.size?.[0] ?? 0;
            const h = this.size?.[1] ?? 0;
            if (!w || !h) return;

            // Skip when collapsed/minimized
            if (this.flags && (this.flags.collapsed || this.flags.minimized)) return;

            const pad = 8;
            const trackH = 6;
            const radius = 3;
            const x = pad;
            // Draw slightly below the node so it never overlaps widgets
            const y = h + 6; // gap below the node
            const trackW = Math.max(0, w - pad * 2);

            // Time-based position (ping-pong 0..1)
            const t = (performance.now() / 1000) * 1.2 + (this.__sjake_phase_offset || 0);
            const pingpong = 0.5 * (1 + Math.sin(t));
            const thumbW = Math.max(16, Math.min(28, trackW * 0.2));
            const thumbX = x + pingpong * (trackW - thumbW);

            // Draw track
            ctx.save();
            ctx.globalAlpha = 0.35;
            roundRect(ctx, x, y, trackW, trackH, radius);
            ctx.fillStyle = "#777a84";
            ctx.fill();
            ctx.restore();

            // Silver thumb with subtle gradient
            const grad = ctx.createLinearGradient(thumbX, y, thumbX + thumbW, y);
            grad.addColorStop(0.0, "#bfc2c9");
            grad.addColorStop(0.5, "#f3f4f7");
            grad.addColorStop(1.0, "#a7aab3");

            // Shadow + fill
            ctx.save();
            ctx.shadowColor = "rgba(0,0,0,0.25)";
            ctx.shadowBlur = 2;
            ctx.shadowOffsetY = 1;
            roundRect(ctx, thumbX, y - 0.5, thumbW, trackH + 1, radius);
            ctx.fillStyle = grad;
            ctx.fill();
            ctx.restore();

            // Top highlight line for a metallic feel
            ctx.save();
            ctx.globalAlpha = 0.35;
            ctx.beginPath();
            ctx.moveTo(thumbX + 1, y + 1);
            ctx.lineTo(thumbX + thumbW - 1, y + 1);
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
        };

        // Lightweight animation loop to keep canvas fresh while node exists
        const ensureRedraw = () => {
            const canvas = app?.canvas;
            if (!canvas) return;
            if (typeof canvas.setDirty === "function") canvas.setDirty(true, true);
            else if (typeof canvas.draw === "function") canvas.draw(true, true);
        };

        const animate = () => {
            if (!node.graph) return; // stop when node is detached
            ensureRedraw();
            node.__sjake_anim_frame = requestAnimationFrame(animate);
        };
        node.__sjake_anim_frame = requestAnimationFrame(animate);

        // Cleanup when removed
        node.onRemoved = function () {
            if (prevOnRemoved) prevOnRemoved();
            if (this.__sjake_anim_frame) cancelAnimationFrame(this.__sjake_anim_frame);
            this.__sjake_anim_frame = null;
        };
    }
});

// Helper: rounded rectangle path
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

