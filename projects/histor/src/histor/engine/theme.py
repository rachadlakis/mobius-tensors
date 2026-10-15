"""Histor's look for the Gradio consoles (histor.ui and the engine's test bench): navy neutrals,
Histor blue as the one accent, quiet small-caps labels, compact spacing, light and dark.

System fonts only: the page loads nothing from the internet (sites can be offline).
"""

from __future__ import annotations

import gradio as gr

# Histor blue (#2864ff) and navy (#0b1230), from the brand kit (Images/histor).
BLUE = gr.themes.Color(c50="#eef3ff", c100="#dbe5ff", c200="#bfd0ff", c300="#93b1ff",
                       c400="#6089ff", c500="#2864ff", c600="#1a4fe6", c700="#163fba",
                       c800="#173896", c900="#182f73", c950="#0b1230", name="histor")
NAVY = gr.themes.Color(c50="#f6f7fb", c100="#eef1f7", c200="#e2e6ef", c300="#c9d0df",
                       c400="#97a1ba", c500="#69738f", c600="#4a5370", c700="#2a3253",
                       c800="#161e3d", c900="#0e1530", c950="#080d1f", name="navy")

LINE_DARK = "rgba(151, 161, 186, 0.16)"  # hairlines on navy

THEME = gr.themes.Base(
    primary_hue=BLUE, secondary_hue=BLUE, neutral_hue=NAVY,
    spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
    font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI Variable Text", "Segoe UI",
          "Roboto", "Helvetica Neue", "sans-serif"],
    font_mono=["ui-monospace", "Cascadia Mono", "Consolas", "SFMono-Regular", "monospace"],
).set(
    body_background_fill="*neutral_50", body_background_fill_dark="*neutral_950",
    body_text_color="*neutral_900", body_text_color_dark="*neutral_100",
    body_text_color_subdued="*neutral_500", body_text_color_subdued_dark="*neutral_400",
    background_fill_primary="white", background_fill_primary_dark="*neutral_900",
    background_fill_secondary="*neutral_50", background_fill_secondary_dark="*neutral_950",
    border_color_primary="*neutral_200", border_color_primary_dark=LINE_DARK,
    color_accent="*primary_500", color_accent_soft="*primary_50",
    color_accent_soft_dark="rgba(40, 100, 255, 0.16)",
    link_text_color="*primary_600", link_text_color_dark="*primary_300",
    # blocks: white cards on a soft grey page; navy cards on near-black
    block_background_fill="white", block_background_fill_dark="*neutral_900",
    block_border_color="*neutral_200", block_border_color_dark=LINE_DARK,
    block_border_width="1px", block_radius="12px", block_padding="10px 12px",
    block_shadow="0 1px 2px rgba(11, 18, 48, 0.04)", block_shadow_dark="none",
    layout_gap="12px",
    # labels: small caps in grey, not filled chips
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_label_border_width="0", block_label_text_color="*neutral_500",
    block_label_text_color_dark="*neutral_400", block_label_text_size="11px",
    block_label_text_weight="600", block_label_padding="6px 8px", block_label_margin="0",
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_border_width="0", block_title_text_color="*neutral_500",
    block_title_text_color_dark="*neutral_400", block_title_text_size="11px",
    block_title_text_weight="600", block_title_padding="0 0 2px",
    block_info_text_size="12px", block_info_text_color="*neutral_500",
    block_info_text_color_dark="*neutral_400",
    panel_background_fill="white", panel_background_fill_dark="*neutral_900",
    panel_border_color="*neutral_200", panel_border_color_dark=LINE_DARK,
    # inputs
    input_background_fill="*neutral_50", input_background_fill_dark="*neutral_950",
    input_background_fill_focus="white", input_background_fill_focus_dark="*neutral_950",
    input_border_color="*neutral_200", input_border_color_dark=LINE_DARK,
    input_border_color_focus="*primary_400", input_border_color_focus_dark="*primary_400",
    input_border_width="1px", input_radius="8px", input_padding="8px 10px",
    input_shadow="none", input_shadow_dark="none",
    input_shadow_focus="0 0 0 3px rgba(40, 100, 255, 0.15)",
    input_shadow_focus_dark="0 0 0 3px rgba(40, 100, 255, 0.25)",
    input_placeholder_color="*neutral_400", input_placeholder_color_dark="*neutral_500",
    checkbox_label_background_fill="transparent", checkbox_label_background_fill_dark="transparent",
    checkbox_label_border_width="0", checkbox_label_padding="4px 0",
    # buttons: one solid blue; everything else quiet
    button_large_radius="9px", button_medium_radius="8px", button_small_radius="7px",
    button_large_padding="9px 18px", button_small_padding="5px 12px",
    button_large_text_size="14px", button_small_text_size="12.5px",
    button_large_text_weight="600", button_small_text_weight="500",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_background_fill_hover_dark="*primary_400",
    button_primary_border_color="*primary_500", button_primary_border_color_dark="*primary_500",
    button_primary_text_color="white", button_primary_text_color_dark="white",
    button_primary_shadow="0 1px 2px rgba(40, 100, 255, 0.30)",
    button_primary_shadow_dark="0 1px 8px rgba(40, 100, 255, 0.35)",
    button_secondary_background_fill="white", button_secondary_background_fill_dark="*neutral_800",
    button_secondary_background_fill_hover="*neutral_100",
    button_secondary_background_fill_hover_dark="*neutral_700",
    button_secondary_border_color="*neutral_200", button_secondary_border_color_dark=LINE_DARK,
    button_secondary_text_color="*neutral_700", button_secondary_text_color_dark="*neutral_100",
    button_cancel_background_fill="white", button_cancel_background_fill_dark="transparent",
    button_cancel_background_fill_hover="#fef2f2",
    button_cancel_background_fill_hover_dark="rgba(239, 68, 68, 0.12)",
    button_cancel_border_color="#fecaca", button_cancel_border_color_dark="rgba(239, 68, 68, 0.4)",
    button_cancel_text_color="#dc2626", button_cancel_text_color_dark="#fca5a5",
    table_border_color="*neutral_200", table_border_color_dark=LINE_DARK,
    table_even_background_fill="white", table_even_background_fill_dark="*neutral_900",
    table_odd_background_fill="*neutral_50", table_odd_background_fill_dark="*neutral_950",
    code_background_fill="*neutral_100", code_background_fill_dark="*neutral_800",
    slider_color="*primary_500", loader_color="*primary_500",
)

CSS = """
.gradio-container {width: 100% !important; max-width: 1320px !important; margin: 0 auto !important;
  padding: 0 20px 24px !important}
.gradio-container .main.fillable {max-width: 100% !important; padding: 0 !important}
* {scrollbar-width: thin; scrollbar-color: var(--neutral-300) transparent}
.dark * {scrollbar-color: var(--neutral-700) transparent}

/* header: icon, name, one line of context, status pills */
.ag-header {display: flex; align-items: center; gap: 12px; padding: 16px 2px 4px}
.ag-header svg {width: 34px; height: 34px; flex: none; border-radius: 9px;
  box-shadow: 0 2px 10px rgba(40, 100, 255, .25)}
.ag-title {font-size: 1.15rem; font-weight: 650; letter-spacing: -.01em; line-height: 1.15}
.ag-sub {color: var(--body-text-color-subdued); font-size: .8rem}
.ag-chips {display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto; justify-content: flex-end}
.ag-chip {display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
  border: 1px solid var(--border-color-primary); border-radius: 999px; font-size: .74rem;
  color: var(--body-text-color-subdued); background: var(--block-background-fill);
  white-space: nowrap}
.ag-chip::before {content: ""; width: 6px; height: 6px; border-radius: 50%; background: #94a3b8}
.ag-chip.on::before {background: #22c55e; box-shadow: 0 0 0 3px rgba(34, 197, 94, .15)}
.ag-chip.plain::before {display: none}
.ag-chip b {color: var(--body-text-color); font-weight: 600}

/* tabs: an underlined nav, not boxes */
.tab-wrapper {margin-bottom: 4px}
[role="tab"] {font-size: .86rem !important; font-weight: 500 !important;
  color: var(--body-text-color-subdued) !important; padding: 9px 12px !important;
  border: 0 !important; background: none !important}
[role="tab"]:hover {color: var(--body-text-color) !important}
[role="tab"][aria-selected="true"] {color: var(--color-accent) !important;
  box-shadow: inset 0 -2px 0 var(--color-accent) !important}
.dark [role="tab"][aria-selected="true"] {color: var(--primary-300) !important;
  box-shadow: inset 0 -2px 0 var(--primary-400) !important}
.ag-subtabs .tab-wrapper {margin-top: 4px}

/* Camera tab: simulate / modules panels, the modules running, the latest events */
/* one card: Gradio's group shows grey between its blocks; here they share the card's fill */
.ag-panel, .ag-panel > .styler, .ag-panel .form, .ag-panel .block {
  background: var(--block-background-fill) !important; border: 0 !important;
  box-shadow: none !important}
.ag-panel {padding: 6px 4px !important; border-radius: 14px !important;
  border: 1px solid var(--border-color-primary) !important; overflow: hidden}
.ag-panel .ag-panel {border: 0 !important; padding: 0 !important}  /* Gradio nests two */
.ag-panel > .styler, .ag-panel .form {gap: 0 !important}
.ag-panel .row {gap: 0 !important}
.ag-panel-title {font-weight: 650; font-size: .95rem; margin-bottom: 4px}
.ag-panel-head .html-container {padding-bottom: 4px !important}
/* buttons and notes inside a card: the card's inner margins */
.ag-panel button.ag-card-btn, .ag-panel .ag-card-btns {margin: 2px 12px 10px !important;
  width: calc(100% - 24px) !important}
.ag-panel .ag-card-btns button {flex: 1}
.ag-panel .ag-card-note .html-container {padding: 0 12px 6px !important}
.ag-panel .ag-card-note .ag-box {margin: 0}
.ag-panel-sub {color: var(--body-text-color-subdued); font-size: .78rem; line-height: 1.45}
/* the drop zone: a quiet dashed area that lights up on hover or drag */
.ag-panel .block.ag-drop {margin: 4px 12px 8px !important; width: auto !important;
  border: 1px dashed rgba(151, 161, 186, .35) !important; border-radius: 10px !important;
  background: rgba(151, 161, 186, .04) !important; transition: border-color .15s, background .15s}
.ag-panel .block.ag-drop:hover, .ag-panel .block.ag-drop:has(.dragging) {
  border-color: var(--primary-400) !important; background: rgba(40, 100, 255, .08) !important}
.ag-drop .upload-container .wrap {font-size: .8rem; color: var(--body-text-color-subdued);
  line-height: 1.35; padding-top: 0 !important}
.ag-drop .upload-container .icon-wrap {color: var(--primary-300); margin-bottom: 4px}
.ag-modules {justify-content: flex-start; margin: 0 0 8px; gap: 6px}
.ag-recent {margin-top: 10px; border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 9px 11px}
.ag-recent > span {color: var(--body-text-color-subdued); font-size: .72rem;
  text-transform: uppercase; letter-spacing: .06em}
.ag-recent ul {margin: 6px 0 0; padding: 0; list-style: none; font-size: .8rem; line-height: 1.6}
.ag-recent li b {font-variant-numeric: tabular-nums; font-weight: 600; margin-right: 4px}
.ag-recent p {margin: 6px 0 0; font-size: .8rem; color: var(--body-text-color-subdued)}

/* the console's main nav: big pill buttons, so the five sections read as the way around */
.ag-nav > .tab-wrapper {margin: 6px 0 14px; padding: 6px; border-radius: 14px;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill)}
.ag-nav > .tab-wrapper [role="tab"] {font-size: 1rem !important; font-weight: 600 !important;
  padding: 11px 22px !important; border-radius: 10px !important; margin-right: 4px;
  min-width: 120px; justify-content: center}
.ag-nav > .tab-wrapper [role="tab"]:hover {background: var(--background-fill-secondary) !important}
.ag-nav > .tab-wrapper [role="tab"][aria-selected="true"] {color: white !important;
  background: var(--color-accent) !important; box-shadow: 0 2px 10px rgba(37, 99, 235, .35) !important}
.dark .ag-nav > .tab-wrapper [role="tab"][aria-selected="true"] {color: white !important;
  background: var(--primary-600) !important}

/* labels on images and galleries: quiet, uppercase; galleries start below theirs */
label[data-testid="block-label"] {text-transform: uppercase; letter-spacing: .06em;
  background: none !important; border: 0 !important; box-shadow: none !important}
.gallery-container .grid-wrap {padding-top: 30px !important}
.ag-screen label[data-testid="block-label"] {background: rgba(5, 8, 26, .55) !important;
  color: #cbd5e1 !important; border-radius: 6px !important; margin: 8px !important}
[data-testid="block-info"] {text-transform: uppercase; letter-spacing: .06em}
.block .info, span[data-testid="block-info"] + div {font-size: 12px}

/* recording status, next to the buttons */
.ag-status {display: inline-flex; align-items: center; gap: 8px; height: 38px; padding: 0 12px;
  border: 1px solid var(--border-color-primary); border-radius: 9px; font-size: .85rem;
  background: var(--block-background-fill); color: var(--body-text-color-subdued);
  max-width: 100%; overflow: hidden; white-space: nowrap; text-overflow: ellipsis}
.ag-dot {width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; flex: none}
.ag-status.rec {color: var(--body-text-color); border-color: rgba(239, 68, 68, .35)}
.ag-status.rec .ag-dot {background: #ef4444; animation: ag-pulse 1.4s infinite}
.ag-status.rec b {color: #ef4444; letter-spacing: .05em; font-size: .75rem}
.ag-status.busy .ag-dot {background: var(--color-accent); animation: ag-pulse 1.4s infinite}
.ag-status.error {color: #b45309; border-color: rgba(245, 158, 11, .4)}
.dark .ag-status.error {color: #fbbf24}
.ag-status.error .ag-dot {background: #f59e0b}
@keyframes ag-pulse {50% {opacity: .3}}

/* key figures */
.ag-cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr)); gap: 8px}
.ag-card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 9px 11px;
  background: var(--block-background-fill)}
.ag-card b {display: block; font-size: 1.35rem; font-weight: 650; line-height: 1.15;
  font-variant-numeric: tabular-nums; letter-spacing: -.01em}
.ag-card span {color: var(--body-text-color-subdued); font-size: .72rem; text-transform: uppercase;
  letter-spacing: .05em}
.ag-notes {margin: 8px 0 0; padding-left: 16px; font-size: .78rem;
  color: var(--body-text-color-subdued)}

/* callouts: tinted, with a coloured dot */
.ag-box {position: relative; border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 9px 12px 9px 30px; font-size: .86rem; background: var(--block-background-fill)}
.ag-box::before {content: ""; position: absolute; left: 12px; top: 14px; width: 8px;
  height: 8px; border-radius: 50%; background: var(--color-accent)}
.ag-box.ok {background: rgba(34, 197, 94, .07); border-color: rgba(34, 197, 94, .3)}
.ag-box.ok::before {background: #22c55e}
.ag-box.warn {background: rgba(245, 158, 11, .07); border-color: rgba(245, 158, 11, .32)}
.ag-box.warn::before {background: #f59e0b}
.ag-box.err {background: rgba(239, 68, 68, .07); border-color: rgba(239, 68, 68, .3)}
.ag-box.err::before {background: #ef4444}
.ag-box.info {background: var(--color-accent-soft); border-color: rgba(40, 100, 255, .22)}
.ag-box p {margin: 3px 0 0; color: var(--body-text-color-subdued); line-height: 1.45}
.ag-box code {font-size: .8em; padding: 1px 4px; border-radius: 4px;
  background: var(--code-background-fill)}
.ag-answer {font-size: .98rem; font-weight: 600}
.ag-muted {color: var(--body-text-color-subdued)}

/* layout helpers */
.ag-field {font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--body-text-color-subdued); white-space: nowrap; padding: 0 2px 0 4px}
.ag-report {padding: 14px 18px !important}
.ag-controls {align-items: center !important; gap: 8px !important}
.ag-controls > * {min-width: 0 !important}
.ag-controls button {height: 38px; flex: 0 0 auto !important; width: auto !important;
  white-space: nowrap; padding: 0 16px !important}
.ag-fieldcell {flex: 0 0 auto !important; width: auto !important}
.ag-side {gap: 8px !important}
.ag-row-tight {gap: 6px !important}
.ag-screen {background: #05081a !important; border-color: transparent !important}
.ag-screen .image-container, .ag-screen .empty {background: #05081a}
.ag-screen .empty svg {opacity: .25}
.ag-examples button {border-radius: 999px !important; font-size: .78rem !important;
  padding: 4px 11px !important}
.ag-examples {gap: 6px}
table th, table th * {font-family: var(--font) !important; font-size: 11px !important;
  font-weight: 600 !important; text-transform: uppercase; letter-spacing: .05em;
  color: var(--body-text-color-subdued) !important}
.prose h1 {font-size: 1.15rem; font-weight: 650}
.prose h2 {font-size: .95rem; font-weight: 650; margin-top: 14px}

/* hand cursor on anything picked from a list */
input[role="combobox"], .wrap:has(> .wrap-inner input[role="combobox"]), .dropdown-arrow,
[role="option"], label:has(> input[type="radio"]), label:has(> input[type="checkbox"]),
input[type="radio"], input[type="checkbox"] {cursor: pointer}

/* an open dropdown list: solid, above what is under it (it was see-through in dark mode) */
ul.option-list {background: var(--background-fill-primary) !important;
  border: 1px solid var(--border-color-primary) !important; border-radius: 10px !important;
  box-shadow: 0 10px 28px rgba(0, 0, 0, .35) !important; padding: 4px !important; z-index: 50}
ul.option-list [role="option"] {border-radius: 7px; padding: 7px 10px !important}
ul.option-list [role="option"]:hover {background: var(--background-fill-secondary) !important}
"""
