#pragma once

#include "PluginProcessor.h"

#include <memory>
#include <vector>

/** Small instrument-panel LookAndFeel — graphite + copper accent. */
class BraveRTLookAndFeel : public juce::LookAndFeel_V4 {
 public:
  BraveRTLookAndFeel();

  void drawRotarySlider(juce::Graphics&, int x, int y, int width, int height,
                        float sliderPosProportional, float rotaryStartAngle,
                        float rotaryEndAngle, juce::Slider&) override;
  void drawLinearSlider(juce::Graphics&, int x, int y, int width, int height,
                        float sliderPos, float minSliderPos, float maxSliderPos,
                        juce::Slider::SliderStyle, juce::Slider&) override;
  void drawToggleButton(juce::Graphics&, juce::ToggleButton&,
                        bool shouldDrawButtonAsHighlighted,
                        bool shouldDrawButtonAsDown) override;
  void drawButtonBackground(juce::Graphics&, juce::Button&,
                            const juce::Colour& backgroundColour,
                            bool shouldDrawButtonAsHighlighted,
                            bool shouldDrawButtonAsDown) override;
  juce::Font getLabelFont(juce::Label&) override;
  juce::Font getTextButtonFont(juce::TextButton&, int buttonHeight) override;
  juce::Font getComboBoxFont(juce::ComboBox&) override;

  static juce::Colour bg() { return juce::Colour(0xff1a1b20); }
  static juce::Colour bgTop() { return juce::Colour(0xff22232a); }
  static juce::Colour accent() { return juce::Colour(0xffd4a574); }
  static juce::Colour accentDim() { return juce::Colour(0xff8a6a4a); }
  static juce::Colour text() { return juce::Colour(0xffe8e6e1); }
  static juce::Colour muted() { return juce::Colour(0xff8a8b94); }
  static juce::Colour warn() { return juce::Colour(0xffc97858); }

  static juce::Font titleFont(float size);
  static juce::Font bodyFont(float size);
};

class BraveRTEditor : public juce::AudioProcessorEditor,
                      private juce::Timer,
                      private juce::ChangeListener {
 public:
  explicit BraveRTEditor(BraveRTProcessor&);
  ~BraveRTEditor() override;

  void paint(juce::Graphics&) override;
  void resized() override;

 private:
  void timerCallback() override;
  void changeListenerCallback(juce::ChangeBroadcaster*) override;

  void rebuildAttrStrip();
  void clearAttrWidgets();
  void syncLoadedChrome(bool animate);
  int computeHeight() const;
  void updateLayoutSize();
  juce::String bundleFolderName() const;

  BraveRTProcessor& proc_;
  BraveRTLookAndFeel lnf_;

  juce::TextButton loadBtn_{"Load bundle"};
  juce::ComboBox blockSizeBox_;
  juce::ToggleButton sidechainBtn_{"Sidechain"};
  juce::Slider gainSlider_;
  juce::Label gainReadout_;
  juce::Label statusWarn_;
  juce::Label emptyAttrLabel_;
  juce::Label condSectionTitle_;
  juce::ComboBox attrModeBox_;  // Stage 8: Manual / Extract / Extract+manual
  juce::Viewport attrViewport_;
  juce::Component attrStrip_;

  struct AttrWidget {
    int slot = -1;
    std::unique_ptr<juce::Label> label;
    std::unique_ptr<juce::Slider> knob;
    std::unique_ptr<juce::ComboBox> menu;
    std::unique_ptr<juce::SliderParameterAttachment> knobAttach;
  };
  std::vector<AttrWidget> attrWidgets_;

  std::unique_ptr<juce::SliderParameterAttachment> gainParamAttach_;
  std::unique_ptr<juce::ComboBoxParameterAttachment> blockSizeAttach_;
  std::unique_ptr<juce::ButtonParameterAttachment> sidechainAttach_;
  std::unique_ptr<juce::ComboBoxParameterAttachment> attrModeAttach_;

  float stripAlpha_ = 1.f;
  float loadFlash_ = 0.f;
  float titleReveal_ = 1.f;     // 0→1 fade/slide of bundle bar content
  float underlineReveal_ = 1.f; // copper underline under transfer title
  float sidechainLed_ = 0.f;
  bool stripFadingIn_ = false;
  bool loadBtnPrimary_ = true;  // copper fill vs outline (swap)

  static constexpr int kWidth = 520;
  static constexpr int kHeaderH = 40;
  static constexpr int kBundleBarH = 36;
  static constexpr int kBaseHeight = 248;  // includes bundle bar
  static constexpr int kAttrRowH = 100;
  static constexpr int kMaxAttrPanelH = 320;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(BraveRTEditor)
};
