// static/js/video_client.js
document.addEventListener("DOMContentLoaded", () => {
    const imgEl = document.getElementById("video-feed");
    const fpsEl = document.getElementById("fps-display");
    const clientsEl = document.getElementById("clients-display");
    const totalEl = document.getElementById("total-frames-display");
    const lastSizeEl = document.getElementById("last-frame-size");

    if (typeof io === "undefined") {
        console.error("Socket.IO no cargó (io undefined)");
        return;
    }

    const socket = io({
        path: "/socket.io",
        transports: ["polling", "websocket"],
        reconnection: true,
        timeout: 20000
    });

    socket.on("connect", () => console.log("✅ WS conectado", socket.id));
    socket.on("connect_error", (e) => console.error("❌ WS error", e?.message || e));
    socket.on("disconnect", (r) => console.log("⚠️ WS disconnect", r));

    socket.on("stats", (data) => {
        if (fpsEl) fpsEl.textContent = (data.fps ?? 0).toFixed(1);
        if (clientsEl) clientsEl.textContent = String(data.connectedClients ?? 0);
        if (totalEl) totalEl.textContent = String(data.totalFrames ?? 0);
    });

    socket.on("new-frame", (p) => {
        if (imgEl && p?.image) imgEl.src = "data:image/jpeg;base64," + p.image;
        if (totalEl && p?.frameNumber != null) totalEl.textContent = String(p.frameNumber);
        if (lastSizeEl && p?.size != null) lastSizeEl.textContent = String(p.size);
    });
});
