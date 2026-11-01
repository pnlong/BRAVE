#include "PluginEditor.h"

// ---------------------------------------------------------------------------
// LookAndFeel
// ---------------------------------------------------------------------------

BraveRTLookAndFeel::BraveRTLookAndFeel() {
  setColour(juce::ResizableWindow::backgroundColourId, bg());
  setColour(juce::Label::textColourId, text());
  setColour(juce::Slider::thumbColourId, accent());
  setColour(juce::Slider::trackColourId, accentDim());
  setColour(juce::Slider::backgroundColourId, juce::Colour(0xff2a2b32));
  setColour(juce::Slider::rotarySliderFillColourId, accent());
  setColour(juce::Slider::rotarySliderOutlineColourId, juce::Colour(0xff3a3b44));
  setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff2a2b32));
  setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff3a3b44));
  setColour(juce::ComboBox::textColourId, text());
  setColour(juce::ComboBox::arrowColourId, accent());
  setColour(juce::PopupMenu::backgroundColourId, bgTop());
  setColour(juce::PopupMenu::textColourId, text());
  setColour(juce::PopupMenu::highlightedBackgroundColourId, accentDim());
  setColour(juce::TextButton::buttonColourId, juce::Colour(0xff2e3038));
  setColour(juce::TextButton::textColourOffId, text());
  setColour(juce::TextButton::textColourOnId, bg());
  setColour(juce::ToggleButton::textColourId, text());
  setColour(juce::ToggleButton::tickColourId, accent());
  setColour(juce::ToggleButton::tickDisabledColourId, muted());
}

juce::Font BraveRTLookAndFeel::titleFont(float size) {
  static const char* kPrefer[] = {"Avenir Next", "Avenir", "Futura", "Gill Sans",
                                  "Trebuchet MS", "Verdana", "DejaVu Sans"};
  for (auto* name : kPrefer) {
    juce::Font trial{juce::FontOptions{}.withName(name).withHeight(size).withStyle(
        "Bold")};
    if (trial.getTypefacePtr() != nullptr &&
        !trial.getTypefaceName().containsIgnoreCase("Default")) {
      trial.setExtraKerningFactor(0.04f);
      return trial;
    }
  }
  juce::Font f{juce::FontOptions(size).withStyle("Bold")};
  f.setExtraKerningFactor(0.04f);
  return f;
}

juce::Font BraveRTLookAndFeel::bodyFont(float size) {
  static const char* kPrefer[] = {"Avenir Next", "Avenir", "Gill Sans",
                                  "Trebuchet MS", "Verdana", "DejaVu Sans"};
  for (auto* name : kPrefer) {
    juce::Font trial{juce::FontOptions{}.withName(name).withHeight(size)};
    if (trial.getTypefacePtr() != nullptr &&
        !trial.getTypefaceName().containsIgnoreCase("Default"))
      return trial;
  }
  return juce::Font{juce::FontOptions(size)};
}

juce::Font BraveRTLookAndFeel::getLabelFont(juce::Label&) {
  return bodyFont(13.f);
}
juce::Font BraveRTLookAndFeel::getTextButtonFont(juce::TextButton&, int) {
  return bodyFont(14.f);
}
juce::Font BraveRTLookAndFeel::getComboBoxFont(juce::ComboBox&) {
  return bodyFont(13.f);
}

void BraveRTLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& b,
                                              const juce::Colour&,
                                              bool highlighted, bool down) {
  auto r = b.getLocalBounds().toFloat().reduced(0.5f);
  // Copper fill only for the empty-state primary CTA ("Load bundle").
  // "Swap bundle…" stays outline so the loaded bar is the focus.
  const bool primary =
      b.getButtonText().equalsIgnoreCase("Load bundle") ||
      b.getProperties().getWithDefault("primary", false);
  auto fill = primary ? accent() : juce::Colour(0xff2e3038);
  if (down) fill = fill.darker(0.15f);
  else if (highlighted) fill = fill.brighter(0.08f);
  g.setColour(fill);
  g.fillRoundedRectangle(r, 5.f);
  if (!primary) {
    g.setColour(highlighted ? accentDim() : juce::Colour(0xff3a3b44));
    g.drawRoundedRectangle(r, 5.f, 1.2f);
  }
}

void BraveRTLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& b,
                                          bool highlighted, bool) {
  auto bounds = b.getLocalBounds().toFloat();
  const float h = 18.f;
  auto track = juce::Rectangle<float>(bounds.getX(), bounds.getCentreY() - h * 0.5f,
                                      34.f, h);
  const bool on = b.getToggleState();
  g.setColour(on ? accentDim() : juce::Colour(0xff2a2b32));
  g.fillRoundedRectangle(track, h * 0.5f);
  auto knobCol = on ? accent() : muted();
  if (highlighted) knobCol = knobCol.brighter(0.1f);
  g.setColour(knobCol);
  const float knob = h - 4.f;
  const float kx = on ? track.getRight() - knob - 2.f : track.getX() + 2.f;
  g.fillEllipse(kx, track.getY() + 2.f, knob, knob);
  g.setColour(text());
  g.setFont(bodyFont(13.f));
  g.drawText(b.getButtonText(),
             bounds.withTrimmedLeft(42.f), juce::Justification::centredLeft, true);
}

void BraveRTLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int w,
                                          int h, float pos, float startAng,
                                          float endAng, juce::Slider&) {
  const float radius = juce::jmin(w, h) * 0.42f;
  const float cx = x + w * 0.5f;
  const float cy = y + h * 0.5f;
  const float ang = startAng + pos * (endAng - startAng);

  juce::Path track;
  track.addCentredArc(cx, cy, radius, radius, 0.f, startAng, endAng, true);
  g.setColour(juce::Colour(0xff2a2b32));
  g.strokePath(track, juce::PathStrokeType(3.5f));

  juce::Path fill;
  fill.addCentredArc(cx, cy, radius, radius, 0.f, startAng, ang, true);
  g.setColour(accent());
  g.strokePath(fill, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved,
                                          juce::PathStrokeType::rounded));

  const float ir = radius * 0.55f;
  g.setColour(bgTop());
  g.fillEllipse(cx - ir, cy - ir, ir * 2.f, ir * 2.f);
  g.setColour(accent());
  const float nx = cx + std::cos(ang - juce::MathConstants<float>::halfPi) * ir * 0.7f;
  const float ny = cy + std::sin(ang - juce::MathConstants<float>::halfPi) * ir * 0.7f;
  g.fillEllipse(nx - 2.5f, ny - 2.5f, 5.f, 5.f);
}

void BraveRTLookAndFeel::drawLinearSlider(juce::Graphics& g, int x, int y, int w,
                                          int h, float pos, float, float,
                                          juce::Slider::SliderStyle style,
                                          juce::Slider& s) {
  if (style != juce::Slider::LinearHorizontal) {
    LookAndFeel_V4::drawLinearSlider(g, x, y, w, h, pos, 0, 0, style, s);
    return;
  }
  const float cy = y + h * 0.5f;
  g.setColour(juce::Colour(0xff2a2b32));
  g.fillRoundedRectangle((float)x, cy - 3.f, (float)w, 6.f, 3.f);
  g.setColour(accent());
  g.fillRoundedRectangle((float)x, cy - 3.f, pos - (float)x, 6.f, 3.f);
  g.setColour(text());
  g.fillEllipse(pos - 7.f, cy - 7.f, 14.f, 14.f);
  g.setColour(accent());
  g.drawEllipse(pos - 7.f, cy - 7.f, 14.f, 14.f, 1.5f);
}

// ---------------------------------------------------------------------------
// Editor
// ---------------------------------------------------------------------------

BraveRTEditor::BraveRTEditor(BraveRTProcessor& p)
    : AudioProcessorEditor(&p), proc_(p) {
  setLookAndFeel(&lnf_);
  proc_.addChangeListener(this);

  addAndMakeVisible(loadBtn_);
  loadBtn_.onClick = [this] {
    auto chooser = std::make_shared<juce::FileChooser>(
        "Select BraveRT bundle folder", juce::File{}, "*");
    chooser->launchAsync(
        juce::FileBrowserComponent::openMode |
            juce::FileBrowserComponent::canSelectDirectories,
        [this, chooser](const juce::FileChooser& fc) {
          auto dir = fc.getResult();
          if (dir.isDirectory()) {
            const bool ok = proc_.loadBundle(dir);
            if (!ok)
              statusWarn_.setText("Load failed — need brave_rt.json + npy_weights/",
                                  juce::dontSendNotification);
          }
        });
  };

  addAndMakeVisible(blockSizeBox_);
  blockSizeBox_.addItemList(BraveRTProcessor::blockSizeLabels(), 1);
  blockSizeAttach_ = std::make_unique<juce::ComboBoxParameterAttachment>(
      *proc_.blockSizeParam, blockSizeBox_);

  addAndMakeVisible(sidechainBtn_);
  sidechainAttach_ = std::make_unique<juce::ButtonParameterAttachment>(
      *proc_.sidechainParam, sidechainBtn_);
  sidechainBtn_.setEnabled(proc_.isCycleGan());

  addAndMakeVisible(gainSlider_);
  gainSlider_.setSliderStyle(juce::Slider::LinearHorizontal);
  gainSlider_.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
  gainParamAttach_ = std::make_unique<juce::SliderParameterAttachment>(
      *proc_.gainParam, gainSlider_);

  addAndMakeVisible(gainReadout_);
  gainReadout_.setJustificationType(juce::Justification::centredRight);
  gainReadout_.setColour(juce::Label::textColourId, BraveRTLookAndFeel::muted());

  addAndMakeVisible(statusWarn_);
  statusWarn_.setJustificationType(juce::Justification::centredLeft);
  statusWarn_.setColour(juce::Label::textColourId, BraveRTLookAndFeel::warn());
  statusWarn_.setFont(BraveRTLookAndFeel::bodyFont(12.f));

  addAndMakeVisible(condSectionTitle_);
  condSectionTitle_.setFont(BraveRTLookAndFeel::bodyFont(13.f));
  condSectionTitle_.setColour(juce::Label::textColourId, BraveRTLookAndFeel::muted());
  condSectionTitle_.setJustificationType(juce::Justification::centredLeft);

  addChildComponent(attrModeBox_);
  attrModeBox_.addItemList(BraveRTProcessor::attrModeLabels(), 1);
  attrModeBox_.setTooltip(
      "Attribute source: Manual = knobs; Extract = rms/centroid/flatness/zcr "
      "from input audio (after ~3 s); Extract+manual = same, reserved for overrides");
  attrModeAttach_ = std::make_unique<juce::ComboBoxParameterAttachment>(
      *proc_.attrModeParam, attrModeBox_);

  addAndMakeVisible(emptyAttrLabel_);
  emptyAttrLabel_.setText("No conditioning features in this bundle",
                          juce::dontSendNotification);
  emptyAttrLabel_.setJustificationType(juce::Justification::centred);
  emptyAttrLabel_.setColour(juce::Label::textColourId, BraveRTLookAndFeel::muted());
  emptyAttrLabel_.setFont(BraveRTLookAndFeel::bodyFont(13.f));

  addAndMakeVisible(attrViewport_);
  attrViewport_.setViewedComponent(&attrStrip_, false);
  attrViewport_.setScrollBarsShown(true, false);
  attrViewport_.getVerticalScrollBar().setColour(
      juce::ScrollBar::thumbColourId, BraveRTLookAndFeel::accentDim());

  rebuildAttrStrip();
  syncLoadedChrome(/*animate=*/false);
  updateLayoutSize();
  startTimerHz(30);
  timerCallback();
}

BraveRTEditor::~BraveRTEditor() {
  proc_.removeChangeListener(this);
  setLookAndFeel(nullptr);
}

juce::String BraveRTEditor::bundleFolderName() const {
  if (!proc_.isBundleLoaded()) return {};
  return juce::File(proc_.bundlePath()).getFileName();
}

void BraveRTEditor::syncLoadedChrome(bool animate) {
  const bool loaded = proc_.isBundleLoaded();
  if (loaded) {
    loadBtn_.setButtonText("Swap bundle…");
    loadBtn_.setTooltip("Load a different BraveRT bundle folder");
    loadBtn_.getProperties().set("primary", false);
    loadBtn_.setColour(juce::TextButton::textColourOffId, BraveRTLookAndFeel::text());
    loadBtnPrimary_ = false;
    if (animate) {
      titleReveal_ = 0.f;
      underlineReveal_ = 0.f;
      loadFlash_ = 1.f;
    } else {
      titleReveal_ = 1.f;
      underlineReveal_ = 1.f;
    }
  } else {
    loadBtn_.setButtonText("Load bundle");
    loadBtn_.setTooltip("Choose a folder with brave_rt.json + npy_weights/");
    loadBtn_.getProperties().set("primary", true);
    // Dark label on copper CTA
    loadBtn_.setColour(juce::TextButton::textColourOffId, BraveRTLookAndFeel::bg());
    loadBtnPrimary_ = true;
    titleReveal_ = 1.f;
    underlineReveal_ = 0.f;
  }
  loadBtn_.repaint();
}

void BraveRTEditor::changeListenerCallback(juce::ChangeBroadcaster*) {
  rebuildAttrStrip();
  const bool flash = proc_.consumeLoadFlash();
  syncLoadedChrome(/*animate=*/flash);
  updateLayoutSize();
  repaint();
}

void BraveRTEditor::clearAttrWidgets() {
  attrWidgets_.clear();
  attrStrip_.removeAllChildren();
}

void BraveRTEditor::rebuildAttrStrip() {
  clearAttrWidgets();
  stripAlpha_ = 0.f;
  stripFadingIn_ = true;

  const auto& hc = proc_.hostControls();
  juce::String section = "Conditioning";
  if (!hc.domain_y_display.empty())
    section = juce::String(hc.domain_y_display);
  condSectionTitle_.setText(section, juce::dontSendNotification);

  const int n = proc_.numVisibleAttrSlots();
  emptyAttrLabel_.setVisible(proc_.isBundleLoaded() && n == 0);
  attrViewport_.setVisible(n > 0);
  sidechainBtn_.setEnabled(proc_.isCycleGan());
  sidechainBtn_.setVisible(proc_.isCycleGan() || !proc_.isBundleLoaded());
  attrModeBox_.setVisible(n > 0 && !proc_.isCycleGan());

  constexpr int cellW = 88;
  constexpr int cellH = kAttrRowH;
  int x = 4;
  for (int slot = 0; slot < n; ++slot) {
    const int si = proc_.attrSlotSchemaIndex(slot);
    if (si < 0 || static_cast<size_t>(si) >= hc.attributes.size()) continue;
    const auto& spec = hc.attributes[static_cast<size_t>(si)];
    AttrWidget w;
    w.slot = slot;
    w.label = std::make_unique<juce::Label>();
    w.label->setText(spec.label.empty() ? spec.name : spec.label,
                     juce::dontSendNotification);
    w.label->setJustificationType(juce::Justification::centred);
    w.label->setColour(juce::Label::textColourId, BraveRTLookAndFeel::muted());
    w.label->setFont(BraveRTLookAndFeel::bodyFont(11.f));
    attrStrip_.addAndMakeVisible(*w.label);

    auto* param = proc_.attrParam(slot);
    const bool isMenu =
        (spec.control == "menu" || spec.control == "choice" ||
         spec.kind == brave_rt::AttrKind::Discrete);

    if (isMenu && param != nullptr) {
      w.menu = std::make_unique<juce::ComboBox>();
      if (!spec.class_labels.empty()) {
        for (size_t i = 0; i < spec.class_labels.size(); ++i)
          w.menu->addItem(spec.class_labels[i], static_cast<int>(i) + 1);
      } else {
        for (int i = 0; i < juce::jmax(1, spec.num_classes); ++i)
          w.menu->addItem(juce::String(i), i + 1);
      }
      const float t = param->get();
      const int idx = juce::roundToInt(proc_.slotNormalizedToRaw(slot, t));
      w.menu->setSelectedItemIndex(juce::jlimit(0, w.menu->getNumItems() - 1, idx),
                                   juce::dontSendNotification);
      w.menu->onChange = [this, slot, menu = w.menu.get()] {
        if (menu == nullptr) return;
        const int i = menu->getSelectedItemIndex();
        const float t01 = proc_.slotRawToNormalized(slot, static_cast<float>(i));
        if (auto* p = proc_.attrParam(slot))
          p->setValueNotifyingHost(p->convertTo0to1(t01));
      };
      attrStrip_.addAndMakeVisible(*w.menu);
    } else if (param != nullptr) {
      w.knob = std::make_unique<juce::Slider>();
      w.knob->setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
      w.knob->setTextBoxStyle(juce::Slider::TextBoxBelow, false, 72, 16);
      w.knob->textFromValueFunction = [this, slot](double) {
        auto* p = proc_.attrParam(slot);
        if (p == nullptr) return juce::String();
        const float raw = proc_.slotNormalizedToRaw(slot, p->get());
        return juce::String(raw, std::abs(raw) >= 100.f ? 0 : 2);
      };
      w.knob->valueFromTextFunction = [this, slot](const juce::String& t) {
        return (double)proc_.slotRawToNormalized(slot, t.getFloatValue());
      };
      w.knobAttach = std::make_unique<juce::SliderParameterAttachment>(
          *param, *w.knob);
      attrStrip_.addAndMakeVisible(*w.knob);
    }

    w.label->setBounds(x, 0, cellW, 16);
    if (w.knob) w.knob->setBounds(x, 16, cellW, cellH - 20);
    if (w.menu) w.menu->setBounds(x + 4, 36, cellW - 8, 24);
    attrWidgets_.push_back(std::move(w));
    x += cellW;
  }
  attrStrip_.setSize(juce::jmax(x, attrViewport_.getWidth()), cellH);
}

int BraveRTEditor::computeHeight() const {
  const int n = proc_.numVisibleAttrSlots();
  const int stripH =
      (!proc_.isBundleLoaded()) ? 40
      : (n == 0)                 ? 48
                                 : juce::jmin(kMaxAttrPanelH, kAttrRowH + 8);
  return kBaseHeight + stripH + 28;
}

void BraveRTEditor::updateLayoutSize() {
  setSize(kWidth, computeHeight());
  resized();
}

void BraveRTEditor::timerCallback() {
  if (stripFadingIn_) {
    stripAlpha_ = juce::jmin(1.f, stripAlpha_ + 0.08f);
    if (stripAlpha_ >= 1.f) stripFadingIn_ = false;
    attrStrip_.setAlpha(stripAlpha_);
    emptyAttrLabel_.setAlpha(stripAlpha_);
  }

  bool animating = false;
  if (loadFlash_ > 0.f) {
    loadFlash_ = juce::jmax(0.f, loadFlash_ - 0.035f);
    animating = true;
  }
  if (titleReveal_ < 1.f) {
    titleReveal_ = juce::jmin(1.f, titleReveal_ + 0.07f);
    animating = true;
  }
  if (proc_.isBundleLoaded() && underlineReveal_ < 1.f) {
    underlineReveal_ = juce::jmin(1.f, underlineReveal_ + 0.05f);
    animating = true;
  }

  if (proc_.isCycleGan() && proc_.sidechainParam != nullptr &&
      proc_.sidechainParam->get()) {
    sidechainLed_ = 0.55f + 0.45f * std::sin(juce::MathConstants<float>::twoPi *
                                             (float)juce::Time::getMillisecondCounter() *
                                             0.001f * 0.7f);
  } else {
    sidechainLed_ = 0.f;
  }

  if (proc_.gainParam != nullptr) {
    const float db = proc_.gainParam->get();
    gainReadout_.setText(juce::String(db, 1) + " dB", juce::dontSendNotification);
  }

  juce::String warn;
  if (!proc_.sampleRateOk())
    warn << "Set DAW to 44.1 kHz (now " << juce::String(proc_.currentSampleRate(), 0)
         << ")";
  if (!proc_.blockSizeMatchesHost() && proc_.hostBlockSize() > 0) {
    if (warn.isNotEmpty()) warn << "  ·  ";
    warn << "Set I/O buffer to " << proc_.selectedBlockSize() << " (host "
         << proc_.hostBlockSize() << ")";
  }
  statusWarn_.setText(warn, juce::dontSendNotification);
  statusWarn_.setVisible(warn.isNotEmpty());

  sidechainBtn_.setEnabled(proc_.isCycleGan());
  if (animating || sidechainLed_ > 0.f)
    repaint(0, 0, getWidth(), kHeaderH + kBundleBarH + 16);
}

void BraveRTEditor::paint(juce::Graphics& g) {
  auto bounds = getLocalBounds().toFloat();
  juce::ColourGradient grad(BraveRTLookAndFeel::bgTop(), 0.f, 0.f,
                            BraveRTLookAndFeel::bg(), 0.f, bounds.getHeight(),
                            false);
  g.setGradientFill(grad);
  g.fillAll();

  // Wordmark
  g.setColour(BraveRTLookAndFeel::text());
  g.setFont(BraveRTLookAndFeel::titleFont(22.f));
  g.drawText("BraveRT", 16, 8, 160, 28, juce::Justification::centredLeft);

  // Sidechain LED (CycleGAN)
  if (proc_.isCycleGan()) {
    const float a = sidechainLed_ > 0.f ? sidechainLed_ : 0.2f;
    g.setColour(BraveRTLookAndFeel::accent().withAlpha(a * 0.9f));
    g.fillEllipse((float)getWidth() - 28.f, 16.f, 10.f, 10.f);
  }

  // Bundle status bar — the main "something is loaded" signal
  auto bar = juce::Rectangle<float>(12.f, (float)kHeaderH,
                                    (float)getWidth() - 24.f, (float)kBundleBarH - 4.f);
  if (proc_.isBundleLoaded()) {
    const float a = titleReveal_;
    g.setColour(juce::Colour(0xff2a2b32).withAlpha(0.95f * a));
    g.fillRoundedRectangle(bar, 6.f);
    // Copper left rail
    g.setColour(BraveRTLookAndFeel::accent().withAlpha(a));
    g.fillRoundedRectangle(bar.getX(), bar.getY() + 4.f, 3.5f, bar.getHeight() - 8.f,
                           1.5f);

    const float slide = (1.f - a) * 8.f;
    auto textArea = bar.reduced(14.f, 4.f).translated(slide, 0.f);

    g.setColour(BraveRTLookAndFeel::accent().withAlpha(0.95f * a));
    g.setFont(BraveRTLookAndFeel::bodyFont(11.f));
    g.drawText("LOADED", textArea.removeFromLeft(52), juce::Justification::centredLeft);

    g.setColour(BraveRTLookAndFeel::text().withAlpha(a));
    g.setFont(BraveRTLookAndFeel::bodyFont(14.f));
    const auto folder = bundleFolderName();
    g.drawText(folder,
               textArea.removeFromLeft(juce::jmin(200.f, textArea.getWidth() * 0.5f)),
               juce::Justification::centredLeft, true);

    if (proc_.modelDisplayName().isNotEmpty()) {
      g.setColour(BraveRTLookAndFeel::muted().withAlpha(a));
      g.setFont(BraveRTLookAndFeel::bodyFont(12.f));
      g.drawText(proc_.modelDisplayName(), textArea, juce::Justification::centredRight,
                 true);
    }

    // Accent underline that draws in under the transfer title / bar
    if (underlineReveal_ > 0.f) {
      const float uw = bar.getWidth() * underlineReveal_;
      g.setColour(BraveRTLookAndFeel::accent().withAlpha(0.55f * a));
      g.fillRect(bar.getX(), bar.getBottom() - 1.5f, uw, 1.5f);
    }
  } else {
    g.setColour(juce::Colour(0xff24252c));
    g.fillRoundedRectangle(bar, 6.f);
    g.setColour(juce::Colour(0xff3a3b44));
    g.drawRoundedRectangle(bar, 6.f, 1.f);
    g.setColour(BraveRTLookAndFeel::muted());
    g.setFont(BraveRTLookAndFeel::bodyFont(13.f));
    g.drawText("No bundle loaded — pick a folder to play", bar.reduced(14.f, 0.f),
               juce::Justification::centredLeft, true);
  }

  // Hairline under chrome
  g.setColour(juce::Colour(0xff2e3038));
  g.fillRect(0, kHeaderH + kBundleBarH, getWidth(), 1);

  // Load success wash
  if (loadFlash_ > 0.f) {
    g.setColour(BraveRTLookAndFeel::accent().withAlpha(0.14f * loadFlash_));
    g.fillRect(getLocalBounds());
  }
}

void BraveRTEditor::resized() {
  auto r = getLocalBounds().reduced(16);
  r.removeFromTop(kHeaderH + kBundleBarH);  // painted header + bundle bar

  auto warnRow = r.removeFromTop(statusWarn_.isVisible() ? 18 : 0);
  statusWarn_.setBounds(warnRow);

  auto row = r.removeFromTop(32);
  loadBtn_.setBounds(row.removeFromLeft(proc_.isBundleLoaded() ? 128 : 120));
  row.removeFromLeft(12);
  blockSizeBox_.setBounds(row.removeFromLeft(72));
  row.removeFromLeft(12);
  if (sidechainBtn_.isVisible())
    sidechainBtn_.setBounds(row.removeFromLeft(110));

  r.removeFromTop(10);
  auto gainRow = r.removeFromTop(28);
  gainReadout_.setBounds(gainRow.removeFromRight(64));
  gainSlider_.setBounds(gainRow);

  r.removeFromTop(12);
  {
    auto titleRow = r.removeFromTop(attrModeBox_.isVisible() ? 22 : 18);
    if (attrModeBox_.isVisible()) {
      attrModeBox_.setBounds(titleRow.removeFromRight(130));
      titleRow.removeFromRight(8);
    }
    condSectionTitle_.setBounds(titleRow);
  }
  r.removeFromTop(4);

  if (!proc_.isBundleLoaded()) {
    emptyAttrLabel_.setText("Load a BraveRT bundle to enable conditioning",
                            juce::dontSendNotification);
    emptyAttrLabel_.setVisible(true);
    emptyAttrLabel_.setBounds(r.removeFromTop(40));
    attrViewport_.setBounds({});
    return;
  }

  emptyAttrLabel_.setText("No conditioning features in this bundle",
                          juce::dontSendNotification);
  const int n = proc_.numVisibleAttrSlots();
  if (n == 0) {
    emptyAttrLabel_.setBounds(r.removeFromTop(40));
    attrViewport_.setBounds({});
  } else {
    emptyAttrLabel_.setBounds({});
    const int vh = juce::jmin(kMaxAttrPanelH, kAttrRowH + 8);
    attrViewport_.setBounds(r.removeFromTop(vh));
    attrStrip_.setSize(juce::jmax(attrStrip_.getWidth(), attrViewport_.getWidth()),
                       kAttrRowH);
  }
}
