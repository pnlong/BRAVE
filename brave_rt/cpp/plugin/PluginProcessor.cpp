#include "PluginProcessor.h"
#include "PluginEditor.h"

BraveRTProcessor::BraveRTProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::mono(), true)
                         .withOutput("Output", juce::AudioChannelSet::mono(), true)) {
  addParameter(bypassParam = new juce::AudioParameterBool(
                   juce::ParameterID{"bypass", 1}, "Bypass", false));
  addParameter(gainParam = new juce::AudioParameterFloat(
                   juce::ParameterID{"gain", 1}, "Output Gain",
                   juce::NormalisableRange<float>(-24.f, 12.f, 0.1f), 0.f,
                   juce::AudioParameterFloatAttributes().withLabel("dB")));
  addParameter(blockSizeParam = new juce::AudioParameterChoice(
                   juce::ParameterID{"block_size", 1}, "Block size",
                   blockSizeLabels(), /*defaultIndex=*/0));
  addParameter(sidechainParam = new juce::AudioParameterBool(
                   juce::ParameterID{"sidechain", 1}, "Sidechain", true));

  addParameter(attrModeParam = new juce::AudioParameterChoice(
                   juce::ParameterID{"attr_mode", 1}, "Attr mode",
                   attrModeLabels(), /*defaultIndex=*/0));

  slotToSchema_.fill(-1);
  for (int i = 0; i < kMaxAttrSlots; ++i) {
    auto* p = new juce::AudioParameterFloat(
        juce::ParameterID{"attr_" + juce::String(i), 1},
        "Attr " + juce::String(i),
        juce::NormalisableRange<float>(0.f, 1.f), 0.5f);
    attrParams_[static_cast<size_t>(i)] = p;
    addParameter(p);
    p->addListener(this);
  }

  blockSizeParam->addListener(this);
  sidechainParam->addListener(this);
  attrModeParam->addListener(this);
}

BraveRTProcessor::~BraveRTProcessor() {
  if (blockSizeParam != nullptr) blockSizeParam->removeListener(this);
  if (sidechainParam != nullptr) sidechainParam->removeListener(this);
  if (attrModeParam != nullptr) attrModeParam->removeListener(this);
  for (auto* p : attrParams_)
    if (p != nullptr) p->removeListener(this);
}

int BraveRTProcessor::selectedBlockSize() const {
  static constexpr int kSizes[] = {128, 256, 512};
  const int i = blockSizeParam != nullptr ? blockSizeParam->getIndex() : 0;
  return kSizes[juce::jlimit(0, 2, i)];
}

void BraveRTProcessor::parameterValueChanged(int parameterIndex, float) {
  // Only block-size changes re-prepare / reset streaming state.
  if (blockSizeParam != nullptr &&
      parameterIndex == blockSizeParam->getParameterIndex()) {
    applyEnginePrepare();
    return;
  }
  if (sidechainParam != nullptr &&
      parameterIndex == sidechainParam->getParameterIndex()) {
    syncSidechainFromParam();
    return;
  }
  syncAttrsFromParams();
}

void BraveRTProcessor::syncSidechainFromParam() {
  if (kind_ == ModelKind::CycleGan && sidechainParam != nullptr)
    cycle_.set_sidechain(sidechainParam->get());
}

float BraveRTProcessor::slotNormalizedToRaw(int slot, float t01) const {
  const int si = attrSlotSchemaIndex(slot);
  if (si < 0 || static_cast<size_t>(si) >= hostControls_.attributes.size())
    return 0.f;
  const auto& s = hostControls_.attributes[static_cast<size_t>(si)];
  t01 = juce::jlimit(0.f, 1.f, t01);
  if (s.kind == brave_rt::AttrKind::Discrete) {
    if (s.num_classes <= 1) return 0.f;
    return static_cast<float>(
        juce::roundToInt(t01 * static_cast<float>(s.num_classes - 1)));
  }
  return s.min_v + t01 * (s.max_v - s.min_v);
}

float BraveRTProcessor::slotRawToNormalized(int slot, float raw) const {
  const int si = attrSlotSchemaIndex(slot);
  if (si < 0 || static_cast<size_t>(si) >= hostControls_.attributes.size())
    return 0.5f;
  const auto& s = hostControls_.attributes[static_cast<size_t>(si)];
  if (s.kind == brave_rt::AttrKind::Discrete) {
    if (s.num_classes <= 1) return 0.f;
    const int idx = juce::jlimit(0, s.num_classes - 1, juce::roundToInt(raw));
    return static_cast<float>(idx) / static_cast<float>(s.num_classes - 1);
  }
  const float span = s.max_v - s.min_v;
  if (std::abs(span) < 1e-12f) return 0.5f;
  return juce::jlimit(0.f, 1.f, (raw - s.min_v) / span);
}

void BraveRTProcessor::setAttrParamNormalized(int slot, float t01) {
  auto* p = attrParam(slot);
  if (p == nullptr) return;
  p->setValueNotifyingHost(p->convertTo0to1(juce::jlimit(0.f, 1.f, t01)));
}

void BraveRTProcessor::syncAttrsFromParams() {
  if (!loaded_) return;
  // Stage 8: attr_mode (Brave/Fader only; CycleGAN engine has no extractor).
  if (kind_ == ModelKind::Brave) brave_.set_attr_mode(hasAttributes() ? engineAttrMode() : 1);
  for (int slot = 0; slot < numVisibleSlots_; ++slot) {
    const int si = slotToSchema_[static_cast<size_t>(slot)];
    if (si < 0) continue;
    auto* p = attrParams_[static_cast<size_t>(slot)];
    if (p == nullptr) continue;
    const float raw = slotNormalizedToRaw(slot, p->get());
    if (kind_ == ModelKind::Brave)
      brave_.set_attr_raw(si, raw);
    else if (kind_ == ModelKind::CycleGan)
      cycle_.set_attr_raw(si, raw);
  }
}

void BraveRTProcessor::rebuildAttrSlotMap() {
  slotToSchema_.fill(-1);
  numVisibleSlots_ = 0;
  hostControls_ = (kind_ == ModelKind::Brave)   ? brave_.host_controls()
                  : (kind_ == ModelKind::CycleGan) ? cycle_.host_controls()
                                                   : brave_rt::HostControls{};

  for (size_t i = 0; i < hostControls_.attributes.size() &&
                     numVisibleSlots_ < kMaxAttrSlots;
       ++i) {
    const auto& a = hostControls_.attributes[i];
    if (!a.applies_to_output) continue;
    const int slot = numVisibleSlots_++;
    slotToSchema_[static_cast<size_t>(slot)] = static_cast<int>(i);
    setAttrParamNormalized(slot, slotRawToNormalized(slot, a.default_v));
  }
  syncAttrsFromParams();
}

void BraveRTProcessor::applyEnginePrepare() {
  const int prep =
      (hostBlock_ > 0) ? hostBlock_ : selectedBlockSize();
  if (kind_ == ModelKind::Brave) {
    brave_.prepare(prep, /*variable_block_sizes=*/false);
    brave_.reset_state();
    syncAttrsFromParams();
    setLatencySamples(kModelLatencySamples + brave_.latency_samples());
  } else if (kind_ == ModelKind::CycleGan) {
    cycle_.prepare(prep, /*variable_block_sizes=*/false);
    cycle_.reset_state();
    syncSidechainFromParam();
    syncAttrsFromParams();
    setLatencySamples(kModelLatencySamples + cycle_.latency_samples());
  } else {
    setLatencySamples(kModelLatencySamples);
  }
}

bool BraveRTProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
  const auto& in = layouts.getMainInputChannelSet();
  const auto& out = layouts.getMainOutputChannelSet();
  if (in != out) return false;
  return in == juce::AudioChannelSet::mono() ||
         in == juce::AudioChannelSet::stereo();
}

void BraveRTProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
  sampleRate_ = sampleRate;
  hostBlock_ = samplesPerBlock;
  sampleRateOk_ = std::abs(sampleRate - 44100.0) <= 1.0;
  if (!sampleRateOk_)
    juce::Logger::writeToLog(
        "BraveRT: expected 44100 Hz (got " + juce::String(sampleRate) + ")");

  mono_in_.assign(static_cast<size_t>(samplesPerBlock), 0.f);
  mono_out_.assign(static_cast<size_t>(samplesPerBlock), 0.f);

  applyEnginePrepare();

  if (!blockSizeMatchesHost())
    juce::Logger::writeToLog(
        "BraveRT: DAW block " + juce::String(hostBlock_) +
        " != plugin Block size " + juce::String(selectedBlockSize()) +
        " — set Logic/Reaper I/O buffer to match for intended RTF");
}

bool BraveRTProcessor::loadBundle(const juce::File& dir) {
  try {
    const auto jsonPath = dir.getChildFile("brave_rt.json");
    if (!jsonPath.existsAsFile())
      throw std::runtime_error("brave_rt.json missing");
    const auto json =
        brave_rt::read_file_text(jsonPath.getFullPathName().toStdString());
    const auto modelType =
        brave_rt::json_string_field(json, "model_type", "brave");
    displayName_ = brave_rt::json_string_field(json, "display_name", "");

    if (modelType == "cyclegan_xy" || modelType == "cyclegan_yx") {
      cycle_.load(dir.getFullPathName().toStdString());
      kind_ = ModelKind::CycleGan;
      if (displayName_.isEmpty())
        displayName_ = cycle_.display_name().empty()
                           ? juce::String(modelType)
                           : juce::String(cycle_.display_name());
    } else if (modelType == "brave" || modelType == "fader") {
      brave_.load(dir.getFullPathName().toStdString());
      kind_ = ModelKind::Brave;
      if (displayName_.isEmpty()) {
        displayName_ = (modelType == "fader") ? "FaderRAVE" : "BRAVE AE";
        const auto& hc = brave_.host_controls();
        if (!hc.domain_y_display.empty())
          displayName_ = juce::String(hc.domain_y_display);
      }
    } else {
      throw std::runtime_error("unsupported model_type: " + modelType);
    }

    rebuildAttrSlotMap();
    applyEnginePrepare();
    bundlePath_ = dir.getFullPathName();
    loaded_ = true;
    loadFlash_ = true;
    sendChangeMessage();
    return true;
  } catch (const std::exception& e) {
    juce::Logger::writeToLog(juce::String("BraveRT load failed: ") + e.what());
    loaded_ = false;
    kind_ = ModelKind::None;
    hostControls_ = {};
    slotToSchema_.fill(-1);
    numVisibleSlots_ = 0;
    sendChangeMessage();
    return false;
  } catch (...) {
    loaded_ = false;
    kind_ = ModelKind::None;
    sendChangeMessage();
    return false;
  }
}

void BraveRTProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                    juce::MidiBuffer&) {
  juce::ScopedNoDenormals noDenormals;
  const int n = buffer.getNumSamples();
  const int nCh = buffer.getNumChannels();

  for (int c = 1; c < nCh; ++c) buffer.clear(c, 0, n);

  if (bypassParam->get() || !loaded_ || kind_ == ModelKind::None) return;

  // Attrs may change from the message thread; cheap float copies.
  syncAttrsFromParams();

  if (static_cast<int>(mono_in_.size()) < n) {
    mono_in_.assign(static_cast<size_t>(n), 0.f);
    mono_out_.assign(static_cast<size_t>(n), 0.f);
  }

  const float* in0 = buffer.getReadPointer(0);
  if (nCh > 1) {
    const float* in1 = buffer.getReadPointer(1);
    for (int i = 0; i < n; ++i)
      mono_in_[static_cast<size_t>(i)] = 0.5f * (in0[i] + in1[i]);
    if (kind_ == ModelKind::Brave)
      brave_.process(mono_in_.data(), mono_out_.data(), n);
    else
      cycle_.process(mono_in_.data(), mono_out_.data(), n);
  } else {
    if (kind_ == ModelKind::Brave)
      brave_.process(in0, mono_out_.data(), n);
    else
      cycle_.process(in0, mono_out_.data(), n);
  }

  const float g =
      juce::Decibels::decibelsToGain(gainParam->get(), -100.f);
  float* out0 = buffer.getWritePointer(0);
  for (int i = 0; i < n; ++i) {
    float y = mono_out_[static_cast<size_t>(i)] * g;
    if (y > 1.f) y = 1.f;
    else if (y < -1.f) y = -1.f;
    out0[i] = y;
    if (nCh > 1) buffer.setSample(1, i, y);
  }
}

juce::AudioProcessorEditor* BraveRTProcessor::createEditor() {
  return new BraveRTEditor(*this);
}

void BraveRTProcessor::getStateInformation(juce::MemoryBlock& destData) {
  juce::XmlElement xml("BraveRTState");
  xml.setAttribute("bundle", bundlePath_);
  xml.setAttribute("bypass", (double)bypassParam->get());
  xml.setAttribute("gain", (double)gainParam->get());
  xml.setAttribute("block_size_index", blockSizeParam->getIndex());
  xml.setAttribute("sidechain", (double)sidechainParam->get());
  xml.setAttribute("attr_mode_index", attrModeParam->getIndex());
  for (int i = 0; i < kMaxAttrSlots; ++i)
    if (attrParams_[static_cast<size_t>(i)] != nullptr)
      xml.setAttribute("attr_" + juce::String(i),
                       (double)attrParams_[static_cast<size_t>(i)]->get());
  copyXmlToBinary(xml, destData);
}

void BraveRTProcessor::setStateInformation(const void* data, int sizeInBytes) {
  auto xml = getXmlFromBinary(data, sizeInBytes);
  if (!xml || !xml->hasTagName("BraveRTState")) return;
  const auto path = xml->getStringAttribute("bundle");
  if (path.isNotEmpty()) loadBundle(juce::File(path));
  *bypassParam = xml->getBoolAttribute("bypass", false);
  *gainParam = (float)xml->getDoubleAttribute("gain", 0.0);
  const int bi = juce::jlimit(0, blockSizeParam->choices.size() - 1,
                              xml->getIntAttribute("block_size_index", 0));
  blockSizeParam->setValueNotifyingHost(
      blockSizeParam->convertTo0to1(static_cast<float>(bi)));
  *sidechainParam = xml->getBoolAttribute("sidechain", true);
  {
    const int ai = juce::jlimit(0, attrModeParam->choices.size() - 1,
                                xml->getIntAttribute("attr_mode_index", 0));
    attrModeParam->setValueNotifyingHost(
        attrModeParam->convertTo0to1(static_cast<float>(ai)));
  }
  for (int i = 0; i < kMaxAttrSlots; ++i) {
    const auto key = "attr_" + juce::String(i);
    if (xml->hasAttribute(key) && attrParams_[static_cast<size_t>(i)] != nullptr)
      setAttrParamNormalized(i, (float)xml->getDoubleAttribute(key, 0.5));
  }
  syncSidechainFromParam();
  syncAttrsFromParams();
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
  return new BraveRTProcessor();
}
