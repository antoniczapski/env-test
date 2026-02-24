"""
test_ipywidgets.py – ipywidgets basic functionality tests.
"""

import pytest


class TestIPyWidgets:
    def test_import(self):
        import ipywidgets  # noqa: F401

    def test_int_slider(self):
        import ipywidgets as widgets

        slider = widgets.IntSlider(value=50, min=0, max=100, step=1)
        assert slider.value == 50
        assert slider.min == 0
        assert slider.max == 100
        slider.value = 75
        assert slider.value == 75

    def test_float_slider(self):
        import ipywidgets as widgets

        slider = widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01)
        assert slider.value == pytest.approx(0.5)

    def test_text_widget(self):
        import ipywidgets as widgets

        text = widgets.Text(value="hello", description="Test:")
        assert text.value == "hello"
        text.value = "world"
        assert text.value == "world"

    def test_dropdown(self):
        import ipywidgets as widgets

        dd = widgets.Dropdown(options=["A", "B", "C"], value="B")
        assert dd.value == "B"

    def test_checkbox(self):
        import ipywidgets as widgets

        cb = widgets.Checkbox(value=True, description="Check")
        assert cb.value is True

    def test_output_widget(self):
        import ipywidgets as widgets

        out = widgets.Output()
        assert out is not None

    def test_hbox_vbox(self):
        import ipywidgets as widgets

        a = widgets.IntSlider()
        b = widgets.IntSlider()
        hbox = widgets.HBox([a, b])
        vbox = widgets.VBox([a, b])
        assert len(hbox.children) == 2
        assert len(vbox.children) == 2

    def test_interactive(self):
        import ipywidgets as widgets
        from ipywidgets import interactive

        def f(x):
            return x ** 2

        w = interactive(f, x=10)
        assert w is not None

    def test_widget_serialization(self):
        """Widgets can be serialized to their widget state dict."""
        import ipywidgets as widgets

        slider = widgets.IntSlider(value=42)
        state = slider.get_state()
        assert "value" in state
        assert state["value"] == 42
