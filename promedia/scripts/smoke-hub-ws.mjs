#!/usr/bin/env node
/**
 * Headless smoke: text + image + video + mediaClock + sceneMode.
 * Hub must listen on :8787. Usage: node scripts/smoke-hub-ws.mjs
 */
const HUB = process.env.HUB_URL ?? "http://127.0.0.1:8787";

function toWsUrl(httpBase, sessionId, role, ticket) {
  const base = httpBase.replace(/^http/, "ws");
  const q = new URLSearchParams({ sessionId, role, credential: ticket });
  return `${base}/api/v1/ws?${q}`;
}

function connect(url) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(url);
    const inbox = [];
    const state = { lastSeq: -1, snapshot: null };
    ws.addEventListener("open", () => {
      resolve({
        ws,
        inbox,
        state,
        get lastSeq() {
          return state.lastSeq;
        },
      });
    });
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(String(ev.data));
      inbox.push(msg);
      if (typeof msg.seq === "number") {
        state.lastSeq = msg.seq;
      }
      if (msg.type === "session.snapshot") {
        state.snapshot = msg.payload;
      }
      if (msg.type === "session.updated") {
        state.snapshot = {
          ...(state.snapshot ?? {}),
          seq: msg.seq,
          live: msg.payload.live,
          playlist: msg.payload.playlist,
        };
      }
    });
    ws.addEventListener("error", () => {
      reject(new Error(`ws error ${url}`));
    });
  });
}

function waitFor(inbox, pred, ms = 4000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const hit = inbox.find(pred);
      if (hit) {
        resolve(hit);
        return;
      }
      if (Date.now() - start > ms) {
        reject(new Error("timeout waiting for message"));
        return;
      }
      setTimeout(tick, 25);
    };
    tick();
  });
}

/** Minimal 1x1 PNG */
const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
  "base64",
);

/** Tiny fake mp4-ish bytes (MIME only; not a real container) */
const FAKE_MP4 = Buffer.from("ftypisomfake-promedia-smoke");

async function main() {
  const createRes = await fetch(`${HUB}/api/v1/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "lan" }),
  });
  if (!createRes.ok) {
    throw new Error(`create ${createRes.status}`);
  }
  const created = await createRes.json();
  console.log("created", created.code, "seq", created.snapshot.seq);
  if (created.snapshot?.live?.mediaClock !== null) {
    throw new Error("expected mediaClock null at create");
  }
  if (created.snapshot?.live?.themeId !== "oscuro") {
    throw new Error(
      `expected default theme oscuro, got ${created.snapshot?.live?.themeId}`,
    );
  }

  const joinOut = await fetch(`${HUB}/api/v1/sessions/join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: created.code, role: "output" }),
  }).then((r) => r.json());

  const control = await connect(
    toWsUrl(HUB, created.sessionId, "control", created.wsTicket),
  );
  const out1 = await connect(
    toWsUrl(HUB, joinOut.sessionId, "output", joinOut.wsTicket),
  );

  await waitFor(control.inbox, (m) => m.type === "session.snapshot");
  await waitFor(out1.inbox, (m) => m.type === "session.snapshot");
  console.log("snapshots ok");

  // text still works
  const textItem = {
    id: "item-text-1",
    type: "text",
    order: 0,
    payload: {
      title: "Smoke",
      align: "center",
      slides: [
        { id: "s1", lines: ["Hello"] },
        { id: "s2", lines: ["World"] },
      ],
    },
  };
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "playlist.itemAdd", payload: { item: textItem } }),
  );
  await waitFor(
    out1.inbox,
    (m) => m.type === "session.updated" && m.payload.playlist?.items?.length === 1,
  );
  console.log("text add ok");

  // upload image
  const form = new FormData();
  form.append(
    "file",
    new Blob([PNG_1X1], { type: "image/png" }),
    "smoke.png",
  );
  const up = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
      body: form,
    },
  );
  if (!up.ok) {
    throw new Error(`upload ${up.status} ${await up.text()}`);
  }
  const { asset } = await up.json();
  console.log("upload ok", asset.id, asset.mime);

  // reject bad mime
  const bad = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
      body: (() => {
        const f = new FormData();
        f.append("file", new Blob([Buffer.from("x")], { type: "text/plain" }), "x.txt");
        return f;
      })(),
    },
  );
  if (bad.status !== 400) {
    throw new Error(`expected mime reject, got ${bad.status}`);
  }
  console.log("mime reject ok");

  const imageItem = {
    id: "item-image-1",
    type: "image",
    order: 1,
    payload: { assetId: asset.id, fit: "contain", title: "smoke.png" },
  };
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "playlist.itemAdd", payload: { item: imageItem } }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.playlist?.items?.some((i) => i.type === "image"),
  );
  console.log("image add ok");

  // edit fit
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "playlist.itemUpdate",
      payload: {
        id: imageItem.id,
        patch: {
          payload: { ...imageItem.payload, fit: "cover" },
        },
      },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.playlist?.items?.find((i) => i.id === imageItem.id)?.payload
        ?.fit === "cover",
  );
  console.log("fit update ok");

  // go live image
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "live.go", payload: { itemId: imageItem.id } }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.currentItemId === imageItem.id &&
      m.payload.live?.mediaClock === null,
  );
  console.log("image go live ok");

  // serve bytes with output ticket
  const get = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets/${asset.id}?credential=${encodeURIComponent(joinOut.wsTicket)}`,
  );
  if (!get.ok) {
    throw new Error(`serve ${get.status}`);
  }
  const buf = Buffer.from(await get.arrayBuffer());
  if (buf.length !== PNG_1X1.length) {
    throw new Error(`serve size mismatch ${buf.length}`);
  }
  console.log("serve ok");

  // path escape / unauthorized
  const noAuth = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets/${asset.id}`,
  );
  if (noAuth.status !== 401) {
    throw new Error(`expected 401 without creds, got ${noAuth.status}`);
  }
  console.log("auth gate ok");

  // --- video file upload ---
  const vform = new FormData();
  vform.append(
    "file",
    new Blob([FAKE_MP4], { type: "video/mp4" }),
    "smoke.mp4",
  );
  const vup = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
      body: vform,
    },
  );
  if (!vup.ok) {
    throw new Error(`video upload ${vup.status} ${await vup.text()}`);
  }
  const { asset: videoAsset } = await vup.json();
  if (videoAsset.kind !== "video" || videoAsset.mime !== "video/mp4") {
    throw new Error(`bad video asset ${JSON.stringify(videoAsset)}`);
  }
  console.log("video upload ok", videoAsset.id);

  // quicktime allowed
  const qt = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
      body: (() => {
        const f = new FormData();
        f.append(
          "file",
          new Blob([Buffer.from("qt")], { type: "video/quicktime" }),
          "clip.mov",
        );
        return f;
      })(),
    },
  );
  if (!qt.ok) {
    throw new Error(`quicktime upload ${qt.status}`);
  }
  console.log("quicktime upload ok");

  // reject unsupported video mime
  const badVid = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
      body: (() => {
        const f = new FormData();
        f.append(
          "file",
          new Blob([Buffer.from("x")], { type: "video/avi" }),
          "x.avi",
        );
        return f;
      })(),
    },
  );
  if (badVid.status !== 400) {
    throw new Error(`expected video mime reject, got ${badVid.status}`);
  }
  console.log("video mime reject ok");

  const videoFileItem = {
    id: "item-video-file-1",
    type: "video",
    order: 2,
    payload: {
      sourceKind: "file",
      assetId: videoAsset.id,
      title: "smoke.mp4",
      autoplay: true,
      muted: false,
      fit: "contain",
    },
  };
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "playlist.itemAdd",
      payload: { item: videoFileItem },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.playlist?.items?.some((i) => i.id === videoFileItem.id),
  );
  console.log("video file add ok");

  // YouTube item
  const ytItem = {
    id: "item-video-yt-1",
    type: "video",
    order: 3,
    payload: {
      sourceKind: "youtube",
      videoId: "dQw4w9WgXcQ",
      sourceUrl: "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
      autoplay: true,
      muted: false,
    },
  };
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "playlist.itemAdd", payload: { item: ytItem } }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.playlist?.items?.some(
        (i) => i.id === ytItem.id && i.payload?.videoId === "dQw4w9WgXcQ",
      ),
  );
  console.log("youtube add ok");

  // invalid youtube rejected by hub validate
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "playlist.itemAdd",
      payload: {
        item: {
          id: "bad-yt",
          type: "video",
          order: 4,
          payload: {
            sourceKind: "youtube",
            videoId: "short",
            sourceUrl: "https://example.com",
          },
        },
      },
    }),
  );
  await waitFor(
    control.inbox,
    (m) => m.type === "error" && m.payload?.code === "invalid_item",
  );
  console.log("invalid youtube reject ok");

  // Go Live video → mediaClock playing true position 0
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "live.go", payload: { itemId: videoFileItem.id } }),
  );
  const liveMsg = await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.currentItemId === videoFileItem.id,
  );
  const clock = liveMsg.payload.live.mediaClock;
  if (
    !clock ||
    clock.itemId !== videoFileItem.id ||
    clock.playing !== true ||
    clock.positionMs !== 0
  ) {
    throw new Error(`bad mediaClock on goLive ${JSON.stringify(clock)}`);
  }
  console.log("video go live clock ok", clock);

  // mediaClock pause
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.mediaClock.set",
      payload: { playing: false, positionMs: 1500 },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.mediaClock?.playing === false &&
      m.payload.live?.mediaClock?.positionMs === 1500,
  );
  console.log("mediaClock pause/seek ok");

  // mediaClock play
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.mediaClock.set",
      payload: { playing: true, positionMs: 2000 },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.mediaClock?.playing === true &&
      m.payload.live?.mediaClock?.positionMs === 2000,
  );
  console.log("mediaClock play ok");

  // second output reconnect sees clock
  const joinOut2 = await fetch(`${HUB}/api/v1/sessions/join`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: created.code, role: "output" }),
  }).then((r) => r.json());
  const out2 = await connect(
    toWsUrl(HUB, joinOut2.sessionId, "output", joinOut2.wsTicket),
  );
  const snap2 = await waitFor(out2.inbox, (m) => m.type === "session.snapshot");
  if (
    snap2.payload.live?.mediaClock?.playing !== true ||
    snap2.payload.live?.mediaClock?.positionMs !== 2000
  ) {
    throw new Error(
      `resync clock mismatch ${JSON.stringify(snap2.payload.live?.mediaClock)}`,
    );
  }
  console.log("output2 resync clock ok");

  // --- scene modes (subtarea 7) ---
  // black: overlay only; keep item + mediaClock playing
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "black" },
    }),
  );
  const blackMsg = await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.sceneMode === "black",
  );
  if (blackMsg.payload.live.currentItemId !== videoFileItem.id) {
    throw new Error("black must keep currentItemId");
  }
  if (blackMsg.payload.live.mediaClock?.playing !== true) {
    throw new Error("black must not auto-pause mediaClock");
  }
  console.log("scene black ok");

  // logo
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "logo" },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" && m.payload.live?.sceneMode === "logo",
  );
  console.log("scene logo ok");

  // clear: sceneMode only, keep item
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "clear" },
    }),
  );
  const clearModeMsg = await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.sceneMode === "clear",
  );
  if (clearModeMsg.payload.live.currentItemId !== videoFileItem.id) {
    throw new Error("clear mode must keep currentItemId");
  }
  if (clearModeMsg.payload.live.mediaClock?.playing !== true) {
    throw new Error("clear must not auto-pause mediaClock");
  }
  console.log("scene clear ok");

  // output2 resync sees clear
  out2.inbox.length = 0;
  out2.ws.send(JSON.stringify({ type: "client.hello", payload: {} }));
  const snapScene = await waitFor(
    out2.inbox,
    (m) => m.type === "session.snapshot",
  );
  if (snapScene.payload.live?.sceneMode !== "clear") {
    throw new Error(
      `resync sceneMode mismatch ${snapScene.payload.live?.sceneMode}`,
    );
  }
  console.log("output2 resync sceneMode ok");

  // back to normal
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "normal" },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.sceneMode === "normal",
  );
  console.log("scene normal ok");

  // invalid sceneMode
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "disco" },
    }),
  );
  await waitFor(
    control.inbox,
    (m) => m.type === "error" && m.payload?.code === "bad_request",
  );
  console.log("invalid sceneMode reject ok");

  // set black then Go Live → resets to normal
  control.ws.send(
    JSON.stringify({
      type: "live.setSceneMode",
      payload: { sceneMode: "black" },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" && m.payload.live?.sceneMode === "black",
  );
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({ type: "live.go", payload: { itemId: textItem.id } }),
  );
  const goReset = await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.currentItemId === textItem.id,
  );
  if (goReset.payload.live.sceneMode !== "normal") {
    throw new Error(
      `Go Live should reset sceneMode to normal, got ${goReset.payload.live.sceneMode}`,
    );
  }
  console.log("goLive resets sceneMode ok");

  // --- themes (subtarea 8) ---
  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setTheme",
      payload: { themeId: "claro" },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.themeId === "claro" &&
      m.payload.live?.themeOverrides === null,
  );
  console.log("theme claro ok");

  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setTheme",
      payload: {
        themeOverrides: { fg: "#ff0000", align: "left", fontSize: "xl" },
      },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.themeId === "claro" &&
      m.payload.live?.themeOverrides?.fg === "#ff0000" &&
      m.payload.live?.themeOverrides?.align === "left",
  );
  console.log("theme overrides ok");

  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setTheme",
      payload: { themeId: "acento" },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.themeId === "acento" &&
      m.payload.live?.themeOverrides === null,
  );
  console.log("theme switch clears overrides ok");

  out1.inbox.length = 0;
  control.ws.send(
    JSON.stringify({
      type: "live.setTheme",
      payload: {
        themeBackground: {
          mode: "media",
          packId: "aurora",
          dotsOverlay: true,
        },
      },
    }),
  );
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.themeBackground?.mode === "media" &&
      m.payload.live?.themeBackground?.packId === "aurora" &&
      m.payload.live?.themeBackground?.dotsOverlay === true,
  );
  console.log("theme background pack + dots ok");

  out2.inbox.length = 0;
  out2.ws.send(JSON.stringify({ type: "client.hello", payload: {} }));
  const snapTheme = await waitFor(
    out2.inbox,
    (m) => m.type === "session.snapshot",
  );
  if (snapTheme.payload.live?.themeId !== "acento") {
    throw new Error(`resync theme ${snapTheme.payload.live?.themeId}`);
  }
  if (snapTheme.payload.live?.themeBackground?.packId !== "aurora") {
    throw new Error(
      `resync themeBackground ${JSON.stringify(snapTheme.payload.live?.themeBackground)}`,
    );
  }
  console.log("output2 resync theme ok");

  // --- saved playlists (subtarea 9) ---
  const savePl = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/playlists`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${created.wsTicket}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ name: "Ensayo culto" }),
    },
  );
  if (!savePl.ok) {
    throw new Error(`save playlist ${savePl.status} ${await savePl.text()}`);
  }
  const savedBody = await savePl.json();
  console.log("playlist save ok", savedBody.playlist.id);

  const listPl = await fetch(`${HUB}/api/v1/playlists`).then((r) => r.json());
  if (!listPl.playlists?.some((p) => p.id === savedBody.playlist.id)) {
    throw new Error("playlist not listed");
  }
  console.log("playlist list ok");

  // load into same session (broadcast)
  out1.inbox.length = 0;
  const loadPl = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/playlists/${savedBody.playlist.id}/load`,
    {
      method: "POST",
      headers: { Authorization: `Bearer ${created.wsTicket}` },
    },
  );
  if (!loadPl.ok) {
    throw new Error(`load playlist ${loadPl.status}`);
  }
  await waitFor(out1.inbox, (m) => m.type === "session.updated");
  console.log("playlist load ok");

  // clear live clears clock + sceneMode
  out1.inbox.length = 0;
  control.ws.send(JSON.stringify({ type: "live.clear", payload: {} }));
  await waitFor(
    out1.inbox,
    (m) =>
      m.type === "session.updated" &&
      m.payload.live?.currentItemId === null &&
      m.payload.live?.mediaClock === null &&
      m.payload.live?.sceneMode === "normal",
  );
  console.log("clear live clock ok");

  // serve video bytes
  const getVid = await fetch(
    `${HUB}/api/v1/sessions/${created.sessionId}/assets/${videoAsset.id}?credential=${encodeURIComponent(joinOut.wsTicket)}`,
  );
  if (!getVid.ok) {
    throw new Error(`serve video ${getVid.status}`);
  }
  console.log("serve video ok");

  control.ws.close();
  out1.ws.close();
  out2.ws.close();
  console.log("SMOKE OK");
}

main().catch((err) => {
  console.error("SMOKE FAIL", err);
  process.exit(1);
});
