(() => {
  const AUDIO_BASE = "assets/audio/";

  const domainSelect = document.getElementById("domain-select");
  const clipSelect = document.getElementById("clip-select");
  const playersRoot = document.getElementById("players");
  const statusEl = document.getElementById("demo-status");

  /** @type {any} */
  let manifest = null;
  /** @type {HTMLAudioElement[]} */
  let audios = [];

  function setStatus(msg) {
    if (statusEl) statusEl.textContent = msg || "";
  }

  function currentDomain() {
    return manifest.domains.find((d) => d.id === domainSelect.value);
  }

  function currentClip(domain) {
    return domain.clips.find((c) => c.id === clipSelect.value) || domain.clips[0];
  }

  function fillSelect(select, items, preferredId) {
    select.innerHTML = "";
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item.id;
      opt.textContent = item.label;
      select.appendChild(opt);
    });
    if (preferredId && items.some((i) => i.id === preferredId)) {
      select.value = preferredId;
    }
  }

  function pauseOthers(active) {
    audios.forEach((audio) => {
      if (audio !== active && !audio.paused) {
        audio.pause();
      }
    });
  }

  function renderPlayers() {
    const domain = currentDomain();
    if (!domain || !domain.clips.length) {
      playersRoot.innerHTML = "<p class='hint'>No clips for this domain.</p>";
      return;
    }

    const clip = currentClip(domain);
    if (clipSelect.value !== clip.id) {
      clipSelect.value = clip.id;
    }

    playersRoot.innerHTML = "";
    audios = [];

    manifest.tracks.forEach((track) => {
      const file = clip.files[track.id];
      if (!file) return;

      const card = document.createElement("article");
      card.className = "player";
      card.dataset.track = track.id;

      const label = document.createElement("div");
      label.className = "label";

      const name = document.createElement("span");
      name.className = "name";
      name.textContent = track.label;

      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = domain.label;

      label.append(name, tag);

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = AUDIO_BASE + file;
      audio.addEventListener("play", () => pauseOthers(audio));

      card.append(label, audio);
      playersRoot.appendChild(card);
      audios.push(audio);
    });

    setStatus(`${domain.label} · ${clip.label}`);
  }

  function onDomainChange() {
    const domain = currentDomain();
    fillSelect(clipSelect, domain.clips, domain.clips[0]?.id);
    renderPlayers();
  }

  async function init() {
    try {
      const res = await fetch(AUDIO_BASE + "manifest.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      manifest = await res.json();
    } catch (err) {
      setStatus("Could not load audio manifest.");
      console.error(err);
      return;
    }

    if (!manifest.domains?.length) {
      setStatus("Manifest has no domains.");
      return;
    }

    fillSelect(domainSelect, manifest.domains, "babbling_brook");
    onDomainChange();

    domainSelect.addEventListener("change", onDomainChange);
    clipSelect.addEventListener("change", renderPlayers);
  }

  init();
})();
