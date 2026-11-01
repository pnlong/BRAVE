#pragma once

#include <juce_audio_processors/juce_audio_processors.h>

#include <array>
#include <vector>

#include "brave_rt/attr_controls.hpp"
#include "brave_rt/brave_engine.hpp"
#include "brave_rt/cyclegan_engine.hpp"
#include "brave_rt/json_util.hpp"

/**
 * BraveRT processor.
 *
 * Stage 7 parameter choice: fixed max slot map (`attr_0` … `attr_15`) as
 * AudioParameterFloat in [0,1]. On loadBundle the slots are remapped to
 * applies_to_output attributes; continuous 0..1 → [min,max], discrete 0..1 →
 * class index. Unused slots stay inert. This keeps automation / state restore
 * stable across bundle swaps (dynamic AudioProcessorParameter trees are
 * awkward in most DAWs).
 *
 * Stage 8 automation surface = those fixed attr slots + ONE `attr_mode`
 * AudioParameterChoice (Manual / Extract / Extract+manual → engine modes
 * 1 / 2 / 0). Manual (default) is the Stage 7 knob-only path. Extract drives
 * rms/centroid/flatness/zcr attrs from the input audio once ~3 s of history
 * exists (before that, and for timbral/discrete attrs, the slot knob/default
 * is used). Extract+manual currently behaves like Extract (no per-attr
 * override params yet). No extra per-attr automation params are created.
 */
class BraveRTProcessor : public juce::AudioProcessor,
                         public juce::ChangeBroadcaster,
                         private juce::AudioProcessorParameter::Listener {
 public:
  enum class ModelKind { None, Brave, CycleGan };

  static constexpr int kMaxAttrSlots = 16;

  BraveRTProcessor();
  ~BraveRTProcessor() override;

  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override {}
  bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
  void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

  juce::AudioProcessorEditor* createEditor() override;
  bool hasEditor() const override { return true; }

  const juce::String getName() const override { return "BraveRT"; }
  bool acceptsMidi() const override { return false; }
  bool producesMidi() const override { return false; }
  double getTailLengthSeconds() const override { return 0.0; }

  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String&) override {}

  void getStateInformation(juce::MemoryBlock& destData) override;
  void setStateInformation(const void* data, int sizeInBytes) override;

  bool loadBundle(const juce::File& dir);
  bool isBundleLoaded() const { return loaded_; }
  juce::String bundlePath() const { return bundlePath_; }
  bool sampleRateOk() const { return sampleRateOk_; }
  double currentSampleRate() const { return sampleRate_; }
  int hostBlockSize() const { return hostBlock_; }
  ModelKind modelKind() const { return kind_; }
  juce::String modelDisplayName() const { return displayName_; }
  bool isCycleGan() const { return kind_ == ModelKind::CycleGan; }
  bool isFader() const { return kind_ == ModelKind::Brave && brave_.num_attributes() > 0; }

  /** Preferred I/O block from the UI choice (128 / 256 / 512). Default 128. */
  int selectedBlockSize() const;
  bool blockSizeMatchesHost() const {
    return hostBlock_ > 0 && hostBlock_ == selectedBlockSize();
  }

  static constexpr int kModelLatencySamples = 128;

  static juce::StringArray blockSizeLabels() { return {"128", "256", "512"}; }

  /** Schema after last successful load (may be empty). */
  const brave_rt::HostControls& hostControls() const { return hostControls_; }
  /** Schema indices for each visible slot (-1 = unused). */
  int attrSlotSchemaIndex(int slot) const {
    return (slot >= 0 && slot < kMaxAttrSlots) ? slotToSchema_[static_cast<size_t>(slot)]
                                               : -1;
  }
  int numVisibleAttrSlots() const { return numVisibleSlots_; }
  juce::AudioParameterFloat* attrParam(int slot) const {
    return (slot >= 0 && slot < kMaxAttrSlots) ? attrParams_[static_cast<size_t>(slot)]
                                               : nullptr;
  }

  /** Map normalized slot value → raw training units. */
  float slotNormalizedToRaw(int slot, float t01) const;
  float slotRawToNormalized(int slot, float raw) const;

  /** Push current attr param values into the active engine. */
  void syncAttrsFromParams();

  /** True after a successful load until editor consumes the flash. */
  bool consumeLoadFlash() {
    const bool f = loadFlash_;
    loadFlash_ = false;
    return f;
  }

  juce::AudioParameterBool* bypassParam = nullptr;
  juce::AudioParameterFloat* gainParam = nullptr;
  juce::AudioParameterChoice* blockSizeParam = nullptr;
  juce::AudioParameterBool* sidechainParam = nullptr;
  juce::AudioParameterChoice* attrModeParam = nullptr;

  /** Choice labels: index 0/1/2 → engine attr_mode 1/2/0. */
  static juce::StringArray attrModeLabels() {
    return {"Manual", "Extract", "Extract+manual"};
  }
  static int attrModeIndexToEngine(int idx) { return idx == 1 ? 2 : (idx == 2 ? 0 : 1); }
  int engineAttrMode() const {
    return attrModeParam != nullptr ? attrModeIndexToEngine(attrModeParam->getIndex()) : 1;
  }
  /** True if the loaded model has attributes (attr_mode only meaningful then). */
  bool hasAttributes() const { return loaded_ && numVisibleSlots_ > 0; }

 private:
  void parameterValueChanged(int parameterIndex, float newValue) override;
  void parameterGestureChanged(int, bool) override {}

  void applyEnginePrepare();
  void syncSidechainFromParam();
  void rebuildAttrSlotMap();
  void setAttrParamNormalized(int slot, float t01);

  brave_rt::BraveEngine brave_;
  brave_rt::CycleGanEngine cycle_;
  brave_rt::HostControls hostControls_;
  std::array<int, kMaxAttrSlots> slotToSchema_{};
  int numVisibleSlots_ = 0;
  std::array<juce::AudioParameterFloat*, kMaxAttrSlots> attrParams_{};

  ModelKind kind_ = ModelKind::None;
  bool loaded_ = false;
  bool loadFlash_ = false;
  bool sampleRateOk_ = true;
  double sampleRate_ = 44100.0;
  int hostBlock_ = 128;
  juce::String bundlePath_;
  juce::String displayName_;
  std::vector<float> mono_in_, mono_out_;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BraveRTProcessor)
};
