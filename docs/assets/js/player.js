(() => {
  const AUDIO_BASE = "assets/audio/";

  const domainSelect = document.getElementById("domain-select");
  const clipSelect = document.getElementById("clip-select");
  const playersRoot = document.getElementById("players");
  const statusEl = document.getElementById("demo-status");
  const sidechainToggle = document.getElementById("sidechain-toggle");
  const sidechainControl = document.getElementById("sidechain-control");

  /** @type {any} */
  let manifest = null;
  /** @type {HTMLAudioElement[]} */
  let audios = [];
  let sidechainOn = true;

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

  function fileForTrack(clip, track) {
    if (track.sidechain && sidechainOn && clip.files[`${track.id}_sc`]) {
      return clip.files[`${track.id}_sc`];
    }
    return clip.files[track.id];
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

    const clipHasSc = Boolean(clip.has_sidechain);
    if (sidechainControl) {
      sidechainControl.hidden = !(manifest.sidechain_available && clipHasSc);
    }
    // Prefer the toggle state when SC stems exist; otherwise force raw.
    const useSc = Boolean(sidechainToggle && sidechainToggle.checked && clipHasSc);
    sidechainOn = useSc;

    playersRoot.innerHTML = "";
    audios = [];

    manifest.tracks.forEach((track) => {
      const file = fileForTrack(clip, track);
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
      tag.textContent =
        track.sidechain && sidechainOn && clip.files[`${track.id}_sc`]
          ? "sidechain on"
          : domain.label;

      label.append(name, tag);

      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "metadata";
      audio.src = AUDIO_BASE + file;
      audio.dataset.trackId = track.id;
      audio.addEventListener("play", () => pauseOthers(audio));

      card.append(label, audio);
      playersRoot.appendChild(card);
      audios.push(audio);
    });

    const scNote =
      sidechainOn && clipHasSc ? " · loudness sidechain on" : " · sidechain off";
    setStatus(`${domain.label} · ${clip.label}${clipHasSc ? scNote : ""}`);
  }

  function onDomainChange() {
    const domain = currentDomain();
    fillSelect(clipSelect, domain.clips, domain.clips[0]?.id);
    renderPlayers();
  }

  function onSidechainToggle() {
    sidechainOn = Boolean(sidechainToggle && sidechainToggle.checked);
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

    sidechainOn = manifest.sidechain_default !== false;
    if (sidechainToggle) {
      sidechainToggle.checked = sidechainOn;
      sidechainToggle.addEventListener("change", onSidechainToggle);
    }

    fillSelect(domainSelect, manifest.domains, "babbling_brook");
    onDomainChange();

    domainSelect.addEventListener("change", onDomainChange);
    clipSelect.addEventListener("change", renderPlayers);
  }

  init();
})();
