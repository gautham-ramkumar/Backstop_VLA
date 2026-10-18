"use strict";

const FPS = 20;
const STARTER_CATEGORIES = [
  "no_grasp_attempt",
  "grasp_failed",
  "misplaced",
  "near_miss",
  "unclassified_no_object_signal",
  "other",
];

const state = {
  shard: null,
  episodes: [],
  index: -1,
};

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const child of children || []) node.appendChild(child);
  return node;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || res.statusText);
  return body;
}

async function loadShards() {
  const shards = await fetchJSON("/api/shards");
  const select = document.getElementById("shard-select");
  select.innerHTML = "";
  for (const s of shards) {
    select.appendChild(el("option", { value: s.shard, text: `${s.shard} (${s.n_failure} failures)` }));
  }
  select.onchange = () => selectShard(select.value, shards);
  if (shards.length) selectShard(shards[0].shard, shards);
}

function updateShardStats(shards) {
  const s = shards.find((x) => x.shard === state.shard);
  const stats = document.getElementById("shard-stats");
  if (!s) { stats.textContent = ""; return; }
  stats.textContent = `${s.n_episodes} episodes, ${s.n_failure} failures, ${s.n_reviewed} reviewed`;
}

// Recomputed from `state.episodes` (already in memory) after a save/reset, so
// the header count updates without a round trip to `/api/shards`.
function refreshStatsFromState() {
  const stats = document.getElementById("shard-stats");
  const nFailure = state.episodes.filter((r) => !r.success).length;
  const nReviewed = state.episodes.filter((r) => r.review.status === "reviewed").length;
  stats.textContent = `${state.episodes.length} episodes, ${nFailure} failures, ${nReviewed} reviewed`;
}

async function selectShard(shard, shards) {
  state.shard = shard;
  updateShardStats(shards);
  state.episodes = await fetchJSON(`/api/episodes?shard=${encodeURIComponent(shard)}`);
  state.index = state.episodes.length ? 0 : -1;
  renderSidebar();
  renderMain();
}

function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = "";
  state.episodes.forEach((row, i) => {
    const outcomeBadge = el("span", {
      class: `badge ${row.success ? "success" : "failure"}`,
      text: row.success ? "OK" : "FAIL",
    });
    const reviewBadge = el("span", {
      class: `badge ${row.review.status}`,
      text: row.review.status === "reviewed" ? "reviewed" : "pending",
    });
    const category = row.review.status === "reviewed" ? row.review.category : row.auto.proposed_category;
    const rowEl = el(
      "div",
      {
        class: "ep-row" + (i === state.index ? " selected" : ""),
        onclick: () => { state.index = i; renderSidebar(); renderMain(); },
      },
      [
        el("span", { class: "ep-idx", text: String(row.episode_index) }),
        outcomeBadge,
        reviewBadge,
        el("span", { class: "ep-cat", text: category || "" }),
      ]
    );
    sidebar.appendChild(rowEl);
  });
}

function fmt(value, digits) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return value.toFixed(digits === undefined ? 3 : digits);
  return String(value);
}

function renderMain() {
  const main = document.getElementById("main");
  main.innerHTML = "";
  if (state.index < 0) {
    main.appendChild(el("div", { id: "empty", text: "No episodes for this shard." }));
    return;
  }
  const row = state.episodes[state.index];

  const video = el("video", {
    controls: "true",
    src: `/api/video?shard=${encodeURIComponent(row.shard)}&episode=${row.episode_index}`,
  });

  const infoPanel = el("div", { class: "panel" }, [
    el("h2", { text: "Episode" }),
    el("div", { id: "task-line", text: row.task }),
    el("dl", { class: "kv" }, [
      el("dt", { text: "shard / episode" }), el("dd", { text: `${row.shard} / ${row.episode_index}` }),
      el("dt", { text: "outcome" }), el("dd", { text: row.success ? "success" : "failure" }),
      el("dt", { text: "n_steps" }), el("dd", { text: String(row.n_steps) }),
      el("dt", { text: "perturbation" }), el("dd", { text: `${row.perturbation} (strength ${row.perturbation_strength})` }),
      el("dt", { text: "params" }), el("dd", { text: JSON.stringify(row.perturbation_params) }),
    ]),
  ]);

  const auto = row.auto;
  const autoPanel = el("div", { class: "panel" }, [
    el("h2", { text: "Auto proposal" }),
    el("dl", { class: "kv" }, [
      el("dt", { text: "category" }), el("dd", { text: auto.proposed_category }),
      el("dt", { text: "onset range" }), el("dd", {
        text: auto.proposed_onset_range ? `[${auto.proposed_onset_range[0]}, ${auto.proposed_onset_range[1]}]` : "n/a",
      }),
      el("dt", { text: "grasped_ever" }), el("dd", { text: String(auto.grasped_ever) }),
      el("dt", { text: "bowl displacement (m)" }), el("dd", { text: fmt(auto.bowl_displacement_m) }),
      el("dt", { text: "terminal bowl-to-plate (m)" }), el("dd", { text: fmt(auto.terminal_bowl_to_plate_m) }),
      el("dt", { text: "eef path length (m)" }), el("dd", { text: fmt(auto.eef_path_length_m) }),
      el("dt", { text: "gripper cycles" }), el("dd", { text: String(auto.gripper_cycle_count) }),
    ]),
  ]);

  const currentCategory = row.review.status === "reviewed" ? row.review.category : auto.proposed_category;
  const currentOnset = row.review.status === "reviewed" ? row.review.onset_range : auto.proposed_onset_range;
  const [onsetStart, onsetEnd] = currentOnset || [0, Math.max(row.n_steps - 1, 0)];

  const categoryInput = el("input", { type: "text", id: "category-input", value: currentCategory || "", list: "category-options" });
  const datalist = el("datalist", { id: "category-options" }, STARTER_CATEGORIES.map((c) => el("option", { value: c })));

  const startInput = el("input", { type: "number", id: "onset-start", min: "0", max: String(row.n_steps - 1), value: String(onsetStart) });
  const endInput = el("input", { type: "number", id: "onset-end", min: "0", max: String(row.n_steps - 1), value: String(onsetEnd) });

  const setFromVideo = (input) => () => {
    const step = Math.round(video.currentTime * FPS);
    input.value = String(Math.max(0, Math.min(row.n_steps - 1, step)));
  };
  const seekTo = (input) => () => {
    video.currentTime = Number(input.value) / FPS;
  };

  const notesInput = el("textarea", { id: "notes-input" });
  notesInput.value = row.review.notes || "";

  const statusLine = el("span", { id: "save-status", text: "" });

  const saveBtn = el("button", {
    type: "button",
    onclick: async () => {
      statusLine.textContent = "saving...";
      try {
        const updated = await fetchJSON("/api/label", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            shard: row.shard,
            episode_index: row.episode_index,
            category: categoryInput.value.trim() || "other",
            onset_range: [Number(startInput.value), Number(endInput.value)],
            notes: notesInput.value,
          }),
        });
        state.episodes[state.index] = updated;
        statusLine.textContent = "saved";
        renderSidebar();
        refreshStatsFromState();
      } catch (err) {
        statusLine.textContent = `error: ${err.message}`;
      }
    },
    text: "Save",
  });

  const resetBtn = el("button", {
    type: "button",
    class: "secondary",
    onclick: async () => {
      const updated = await fetchJSON("/api/label/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ shard: row.shard, episode_index: row.episode_index }),
      });
      state.episodes[state.index] = updated;
      renderSidebar();
      renderMain();
      refreshStatsFromState();
    },
    text: "Reopen (pending)",
  });

  const prevBtn = el("button", {
    type: "button",
    class: "secondary",
    onclick: () => { if (state.index > 0) { state.index -= 1; renderSidebar(); renderMain(); } },
    text: "< Prev",
  });
  const nextBtn = el("button", {
    type: "button",
    class: "secondary",
    onclick: () => { if (state.index < state.episodes.length - 1) { state.index += 1; renderSidebar(); renderMain(); } },
    text: "Next >",
  });

  const reviewForm = el("form", { id: "review-form", onsubmit: (e) => e.preventDefault() }, [
    el("div", {}, [el("label", { text: "Category (free text -- codebook not frozen yet)" }), categoryInput, datalist]),
    el("div", {}, [
      el("label", { text: "Onset range (steps)" }),
      el("div", { class: "onset-row" }, [
        startInput,
        el("button", { type: "button", class: "tiny secondary", onclick: setFromVideo(startInput), text: "set start" }),
        el("button", { type: "button", class: "tiny secondary", onclick: seekTo(startInput), text: "seek" }),
      ]),
      el("div", { class: "onset-row" }, [
        endInput,
        el("button", { type: "button", class: "tiny secondary", onclick: setFromVideo(endInput), text: "set end" }),
        el("button", { type: "button", class: "tiny secondary", onclick: seekTo(endInput), text: "seek" }),
      ]),
    ]),
    el("div", {}, [el("label", { text: "Notes" }), notesInput]),
    el("div", { class: "row" }, [saveBtn, resetBtn, statusLine]),
  ]);

  const reviewPanel = el("div", { class: "panel" }, [el("h2", { text: "Review" }), reviewForm]);

  const nav = el("div", { class: "row" }, [prevBtn, nextBtn]);

  const panels = el("div", { id: "panels" }, [infoPanel, autoPanel, reviewPanel, nav]);
  const videoWrap = el("div", { id: "video-wrap" }, [video, panels]);
  main.appendChild(videoWrap);
}

loadShards().catch((err) => {
  document.getElementById("main").textContent = `Failed to load: ${err.message}`;
});
