"use strict";

const state = { index: [], shard: null, signal: null, episode: null };

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children || []) node.appendChild(child);
  return node;
}

async function fetchJSON(url) {
  const response = await fetch(url);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

function episodes() {
  return state.index.filter((row) => row.shard === state.shard);
}

function renderSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.innerHTML = "";
  for (const row of episodes()) {
    const selected = state.episode === row.episode_index ? " selected" : "";
    sidebar.appendChild(el("div", {
      class: "ep" + selected,
      onclick: () => { state.episode = row.episode_index; renderSidebar(); showEpisode(); },
    }, [
      el("span", { class: row.success ? "ok" : "fail", text: row.success ? "success " : "failure " }),
      el("span", { text: String(row.episode_index) }),
    ]));
  }
}

function polyline(values) {
  if (!values || values.length < 2) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return values.map((value, i) => {
    const x = (i / (values.length - 1)) * 640;
    const y = 150 - ((value - min) / span) * 140;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

async function showEpisode() {
  const row = episodes().find((item) => item.episode_index === state.episode);
  const main = document.getElementById("main");
  if (!row) {
    main.textContent = "Pick an episode.";
    return;
  }
  const trace = await fetchJSON(`/api/trace?shard=${encodeURIComponent(state.shard)}&episode=${state.episode}`);
  const series = trace.per_step[state.signal] || [];
  const score = trace.scores[state.signal];
  main.innerHTML = "";
  const video = el("video", { controls: "true", src: `/api/video?shard=${encodeURIComponent(state.shard)}&episode=${state.episode}` });
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 640 160");
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "#5b8cff");
  line.setAttribute("stroke-width", "2");
  line.setAttribute("points", polyline(series));
  svg.appendChild(line);
  main.appendChild(video);
  main.appendChild(svg);
  main.appendChild(el("p", { id: "meta", text: `${trace.task} — ${state.signal} episode score ${score == null ? "n/a" : Number(score).toFixed(3)}, trust ${trace.trust == null ? "n/a" : Number(trace.trust).toFixed(3)}` }));
}

async function boot() {
  state.index = await fetchJSON("/api/index");
  const shards = [...new Set(state.index.map((row) => row.shard))];
  const shardSelect = document.getElementById("shard");
  for (const shard of shards) shardSelect.appendChild(el("option", { value: shard, text: shard }));
  const names = state.index.length ? Object.keys(state.index[0].scores) : [];
  const signalSelect = document.getElementById("signal");
  for (const name of names) signalSelect.appendChild(el("option", { value: name, text: name }));
  state.shard = shards[0] || null;
  state.signal = names[0] || null;
  state.episode = episodes()[0] ? episodes()[0].episode_index : null;
  shardSelect.onchange = () => { state.shard = shardSelect.value; state.episode = episodes()[0]?.episode_index; renderSidebar(); showEpisode(); };
  signalSelect.onchange = () => { state.signal = signalSelect.value; showEpisode(); };
  document.getElementById("meta").textContent = `${state.index.length} episodes`;
  renderSidebar();
  if (state.episode != null) showEpisode();
}

boot().catch((error) => { document.getElementById("main").textContent = error.message; });
