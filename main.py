from manim import *
import numpy as np
import math
import os
from typing import Tuple, List, Dict

###############################################################################
# consts & simple data classess
###############################################################################

T: float = 10.0           # total time‑span (s)
DT: float = 0.01          # simulation step (s)
FIRING_RATE: float = 5.0  # spikes/s
SPIKE_HEIGHT: float = 3.0
AXES_LENGTH: Tuple[float, float] = (13.3, 2.85)
DEFAULT_X_RANGE = (0, T, 1)


class Ctx:
    """Singleton‑like context for passing state between scenes."""

    ## spike train
    time_points: np.ndarray | None = None
    spikes: np.ndarray | None = None

    # discrete bin config
    last_bin_count: int | None = None
    last_bin_bars: VGroup | None = None
    last_bin_axes_cfg: Dict | None = None


###############################################################################
# helpers
###############################################################################


def create_axes(
    x_range: Tuple[float, float, float] = DEFAULT_X_RANGE,
    y_range: Tuple[float, float, float] = (0, 6, 1),
    length: Tuple[float, float] = AXES_LENGTH,
    label_x: str | None = None,
    label_y: str | None = None,
) -> Axes:
    """Factory for axes with optional labels."""
    x_len, y_len = length
    axes = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=x_len,
        y_length=y_len,
        tips=False,
    )
    if label_x:
        axes.add(axes.get_x_axis_label(label_x, direction=DOWN))
    if label_y:
        axes.add(axes.get_y_axis_label(label_y))
    return axes


def simulate_spike_train(
    T_: float = T,
    dt_: float = DT,
    rate: float = FIRING_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time_points, boolean spikes)."""
    t = np.arange(0, T_, dt_)
    spikes = np.random.rand(len(t)) < rate * dt_
    return t, spikes


def spike_curve(
    axes: Axes,
    t: np.ndarray,
    spikes: np.ndarray,
    height: float = SPIKE_HEIGHT,
    collapsed: bool = False,
    color=YELLOW,
) -> VMobject:
    """Return a VMobject polyline representing the spike train."""
    pts: List[np.ndarray] = []
    for tt, spk in zip(t, spikes):
        p0 = axes.c2p(tt, 0)
        if spk:
            if collapsed:
                pts.extend([p0, p0, p0])
            else:
                pts.extend([p0, axes.c2p(tt, height), p0])
        else:
            pts.append(p0)
    return VMobject(color=color).set_points_as_corners(pts)


def create_binned_visualisation(
    axes: Axes,
    t: np.ndarray,
    spikes: np.ndarray,
    num_bins: int,
    dt_: float = DT,
    spike_height: float = SPIKE_HEIGHT,
) -> Tuple[VGroup, VGroup, VGroup]:
    """Return (bin_lines, bin_containers, bars)."""
    bins = np.linspace(0, T, num_bins + 1)
    # count of spikes
    counts = [
        int(np.sum(spikes[int(bins[i] / dt_) : int(bins[i + 1] / dt_)]))
        for i in range(num_bins)
    ]

    # creating the visualisation
    bin_lines = VGroup()
    containers = VGroup()
    bars = VGroup()
    y_max = axes.y_range[1]

    for i in range(num_bins):
        x0, x1 = bins[i], bins[i + 1]
        left, right = axes.c2p(x0, 0), axes.c2p(x1, 0)
        # sub div line
        bin_lines.add(DashedLine(start=left, end=axes.c2p(x0, y_max), dash_length=0.1, color=GREY_C))
        # the "bucket" container
        depth = 0.5
        containers.add(
            VGroup(
                Line(left, right),
                Line(left, [left[0], left[1] - depth, 0]),
                Line(right, [right[0], right[1] - depth, 0]),
                Line([left[0], left[1] - depth, 0], [right[0], right[1] - depth, 0]),
            )
        )
        # bar
        bar_width = (right[0] - left[0]) - 0.05
        h = max(counts[i] * spike_height / 6, 0.01)
        bar = Rectangle(
            width=bar_width,
            height=h,
            fill_color=BLUE,
            fill_opacity=0.7,
            stroke_color=WHITE,
            stroke_width=1,
        )
        bar.align_to(axes.c2p(x0 + 0.025, 0), LEFT + DOWN)
        bars.add(bar)

    # final line to make bins closed
    bin_lines.add(DashedLine(start=axes.c2p(T, 0), end=axes.c2p(T, y_max), dash_length=0.1, color=GREY_C))

    return bin_lines, containers, bars


# -----------------------------------------------------------------------------
# very rouhg Izhikevich neuron sim (regular‑spiking variant)
# -----------------------------------------------------------------------------


def simulate_izhikevich(total_ms: float = 400.0, dt_ms: float = 0.1, inj_pA: float = 70.0):

    # Izhikevich regular‑spiking parameters
    a, b, c, d = 0.02, 0.2, -65.0, 8.0

    steps = int(total_ms / dt_ms)
    v = -65.0
    u = b * v

    v_trace = np.zeros(steps)

    I = np.zeros(steps)
    t_on  = int(100 / dt_ms)
    t_off = int(200 / dt_ms)
    I[t_on:t_off] = inj_pA

    for t in range(steps):
        if v >= 30.0: # spike threshold = 30
            v_trace[t] = 30.0
            v = c
            u += d
        else:
            v_trace[t] = v

        dv = (0.04 * v + 5.0) * v + 140.0 - u + I[t]
        v += dv * dt_ms
        du = a * (b * v - u)
        u += du * dt_ms

    return v_trace, dt_ms

# for caching the voltage trace
_CACHE_FILE = "izhikevich_voltage.npz"


def get_voltage_trace(overwrite: bool = False):
    if os.path.exists(_CACHE_FILE) and not overwrite:
        with np.load(_CACHE_FILE, allow_pickle=True) as data:
            v_trace = data['v_trace']
            dt_ms = float(data['dt_ms'])
        return v_trace, dt_ms
    v_trace, dt_ms = simulate_izhikevich()
    np.savez(_CACHE_FILE, v_trace=v_trace, dt_ms=dt_ms)
    return v_trace, dt_ms

###############################################################################
# Scene mixin with common set‑up                                              #
###############################################################################


class SpikeScene(Scene):
    """Base‑class providing axes + synthetic spike train."""

    axes: Axes
    time_points: np.ndarray
    spikes: np.ndarray

    def setup_axes(self, y_label: str = "Amplitude") -> None:
        self.axes = create_axes(label_x="Time (s)", label_y=y_label)
        self.axes.scale(0.95).to_edge(DOWN)
        self.add(self.axes)

    def setup_spikes(self) -> None:
        if Ctx.time_points is None or Ctx.spikes is None:
            Ctx.time_points, Ctx.spikes = simulate_spike_train()
        self.time_points = Ctx.time_points
        self.spikes = Ctx.spikes

###############################################################################
# Individual scenes                                                           #
###############################################################################


class ReasonsForDirac(Scene):
    """Compare a digital square wave with a realistic spiking‑neuron trace."""

    def construct(self):
        # title
        title = (
            Text("Neurons behave very differently from the binary nature of computers.")
            .scale(0.6)
            .to_edge(UP)
        )
        self.play(Write(title))
        self.wait(0.5)


        bit_axes = create_axes(
            y_range=(0, 2, 1), length=(5, 3), label_x="t", label_y="V (bits)"
        ).to_corner(LEFT + DOWN)

        # graph of digital oscillator
        subtitle = Text("Digital circuits are very robust, \n and only oscillate between two values").scale(0.4).next_to(title, DOWN+1)
        # subtitle above the axes
        subtitle.move_to(bit_axes.get_top() + 0.5 * UP)
        self.play(Write(subtitle))

        self.play(Create(bit_axes))

        square = bit_axes.plot(lambda x: 2 if int(x) % 2 == 0 else 0,
                               x_range=[0, 8, 0.01], color=YELLOW)
        self.play(Create(square))

        # neuron mv graph - using Izhikevich model
        neuron_axes = create_axes(
            x_range=(0, 400, 50),
            y_range=(-80, 40, 20),
            length=(5, 3),
            label_x="t (ms)",
            label_y="V (mV)",
        ).to_corner(RIGHT + DOWN)

        subtitle = Text("Biological neurons are much more complex, and have a continuous range of voltages").scale(0.4).next_to(title, DOWN)
        subtitle.move_to(neuron_axes.get_top() + 0.5 * UP)
        self.play(Write(subtitle))

        self.play(Create(neuron_axes))

        lif_eq = (
            MathTex(r"\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L}")
            .scale(0.5)
            .next_to(subtitle, DOWN, buff=0.3)
        )
        self.play(Write(lif_eq))

        # sim
        self.wait(0.5)
        voltage, dt_ms = get_voltage_trace()
        time_ms = np.arange(len(voltage)) * dt_ms

        # we want manim pionts
        points = [neuron_axes.c2p(t, v) for t, v in zip(time_ms, voltage)]
        trace = VMobject(color=YELLOW).set_points_smoothly(points)
        self.play(Create(trace), run_time=6)
        self.wait(1)


        # new part: focus on the neuron graph, section is about the goal of neuron graphs: finding the spikes and firing rate
        title2 = Text("The goal of modeling neuron activity is to find the spikes and firing rate", font_size=36).to_edge(UP)

        # fade out the old stuff, move graph to the center
        self.play(
            FadeOut(bit_axes),
            FadeOut(square),
            FadeOut(lif_eq),
            FadeOut(subtitle),
            FadeOut(title),
            # move the neuron graph to the center
            # this is the axis, trade the x and y axis
            neuron_axes.animate.move_to(ORIGIN).scale(1.5),
            trace.animate.move_to(ORIGIN).scale(1.5),
            # make this animation 1 second
            run_time=1,
        )

        self.play(Write(title2))



class DiracDeltaApproximation(Scene):
    """gossly simplified but visually intuitive Dirac‑delta derivation."""

    def construct(self):
        eq = MathTex(r"\delta(x)=\begin{cases}0 & x\neq0 \\ \infty & x=0\end{cases},\;\int\!\delta(x)\,dx=1").scale(0.8).to_edge(UP)
        title = Text("The Dirac Delta Function", font_size=36).next_to(eq, DOWN)
        self.play(Write(eq), Write(title))
        self.wait()
        self.play(FadeOut(eq), FadeOut(title))

        axes = create_axes(x_range=(-5, 5, 1), y_range=(0, 10, 1), length=(10, 6), label_x="x", label_y="y").to_edge(DOWN)
        self.play(Create(axes))

        a = ValueTracker(1.0)

        def lorentz(x: float) -> float:
            aa = max(a.get_value(), 0.01)
            return (1 / (math.pi * aa)) * (1 / (1 + (x / aa) ** 2))

        graph = always_redraw(lambda: axes.plot(lorentz, x_range=[-5, 5, 0.01], color=BLUE))
        area = always_redraw(lambda: axes.get_area(graph, x_range=[-5, 5], color=BLUE_B, opacity=0.3))
        label_a = always_redraw(lambda: MathTex(f"a={a.get_value():.2f}").set_color(YELLOW).to_corner(UL))
        self.play(Create(graph), Create(area), Write(label_a))
        self.play(a.animate.set_value(0.05), run_time=5, rate_func=rate_functions.ease_in_cubic)

        vline = axes.get_vertical_line(axes.c2p(0, axes.y_range[1]), color=RED, stroke_width=3)
        vline.set_opacity(0)
        infinity = MathTex("\infty", color=RED).next_to(vline, UP)
        infinity.set_opacity(0)

        self.play(graph.animate.set_opacity(0.5), area.animate.set_opacity(0.15))
        self.play(vline.animate.set_opacity(1), infinity.animate.set_opacity(1))
        self.wait(1)


class DiscreteBinGraph(SpikeScene):
    """bins the spike train and shows histogram‑like bars."""

    def construct(self):
        self.setup_axes()
        self.setup_spikes()

        title = Text("Instantaneous spikes ≈ Dirac deltas", font_size=28).to_edge(UP)
        self.play(Write(title))

        collapsed = spike_curve(self.axes, self.time_points, self.spikes, collapsed=True)
        curve = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(collapsed))
        self.play(Transform(collapsed, curve, run_time=3))

        # ------- discrete bins -------
        bin_lines, containers, bars = create_binned_visualisation(
            self.axes, self.time_points, self.spikes, num_bins=15
        )

        # record for next scene
        Ctx.last_bin_count = 15
        Ctx.last_bin_axes_cfg = dict(x_range=DEFAULT_X_RANGE, y_range=(0, 6, 1))
        Ctx.last_bin_bars = bars

        explanation = Text("Visualising by grouping spikes into equal bins", font_size=24).next_to(title, DOWN, buff=0.4)
        self.play(Write(explanation))
        self.play(LaggedStart(*[Create(line) for line in bin_lines], lag_ratio=0.05))
        self.play(LaggedStart(*[Create(cont) for cont in containers], lag_ratio=0.05))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05))
        self.wait()

        # demonstrate different bin counts
        for n in (5, 10, 20, 30):
            nl, nc, nb = create_binned_visualisation(self.axes, self.time_points, self.spikes, n)
            txt = Text(f"Bins: {n}", font_size=24, color=YELLOW).to_edge(RIGHT)
            self.play(
                FadeOut(bin_lines), FadeOut(containers), FadeOut(bars), Write(txt)
            )
            self.play(LaggedStart(*[Create(l) for l in nl], lag_ratio=0.05))
            self.play(LaggedStart(*[Create(c) for c in nc], lag_ratio=0.05))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in nb], lag_ratio=0.05))
            self.wait(1)
            self.play(FadeOut(txt))
            bin_lines, containers, bars = nl, nc, nb
        self.wait()


class SlidingWindowApprox(SpikeScene):
    """Starts with bars from previous scene, fades them, then sliding rectangular window."""

    def construct(self):
        self.setup_axes("Count")
        self.setup_spikes()

        # prev recreation
        if Ctx.last_bin_count:
            l, c, b = create_binned_visualisation(
                self.axes, self.time_points, self.spikes, Ctx.last_bin_count
            )
            self.add(*l, *c, *b)
            self.play(FadeOut(VGroup(*l, *c, *b), run_time=1))

        # spike trace
        trace = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(trace))

        # sliding window rectangle + live counter
        win = Rectangle(
            width=self.axes.x_axis.unit_size * 0.5,
            height=self.axes.y_length,
            stroke_color=GREY,
            stroke_width=2,
            fill_color=BLUE,
            fill_opacity=0.5,
        )
        tracker = ValueTracker(0.25)
        win.add_updater(lambda m: m.move_to(self.axes.c2p(tracker.get_value(), SPIKE_HEIGHT / 2)))
        self.add(win)

        # pre‑compute counts within 0.5‑s window
        half = 0.25
        centres = np.arange(half, T - half + 1e-6, DT)
        counts = [np.sum(self.spikes[int((c - half) / DT): int((c + half) / DT)]) for c in centres]
        count_curve = VMobject(color=RED, stroke_width=2).set_points_as_corners([
            self.axes.c2p(c, cnt) for c, cnt in zip(centres, counts)
        ]).stretch(0.5, 1, about_point=self.axes.c2p(0, 0))

        counter = Integer(0).next_to(win, UP, buff=0.1)
        counter.add_updater(
            lambda m: m.set_value(counts[int((tracker.get_value() - half) / DT)]).next_to(win, UP, buff=0.1)
        )
        self.add(counter)

        self.play(tracker.animate.set_value(T - half), Create(count_curve), run_time=8, rate_func=linear)
        self.play(FadeOut(VGroup(trace, win, counter)))
        self.wait()




class GaussianWindow(SpikeScene):
    """Smooths the rectangular sliding‑window estimate with a Gaussian kernel."""

    def construct(self):
        self.setup_axes("Count")
        self.setup_spikes()

        # why we need a gaussian
        expl1 = Text(
            "The rectangular sliding window still gives a blocky, discontinuous estimate.",
            font_size=26
        ).to_edge(UP)
        self.play(Write(expl1))

        expl2 = Text(
            "→ Replace it with a smooth *Gaussian* window:", font_size=26, slant=ITALIC
        ).next_to(expl1, DOWN, buff=0.3)
        self.play(Write(expl2))

        gauss_eq = MathTex(r"w(t)=\frac{1}{\sqrt{2\pi}\,\sigma}\,e^{-t^{2}/(2\sigma^{2})}").scale(0.8)
        gauss_eq.next_to(expl2, DOWN, buff=0.3)
        self.play(Write(gauss_eq))

        # old rectangle
        rect_width = 0.5
        rect = Rectangle(
            width=self.axes.x_axis.unit_size * rect_width,
            height=self.axes.y_length,
            stroke_color=GREY,
            stroke_width=2,
            fill_color=BLUE,
            fill_opacity=0.5,
        )
        rect.next_to(gauss_eq, LEFT, buff=0.8)
        self.play(Create(rect))

        # gaussian helper
        sigma = 0.25
        tracker = ValueTracker(rect_width / 2)  # will be reset later

        def bell_path(center: float):
            return self.axes.plot(
                lambda x: math.exp(-((x - center) ** 2) / (2 * sigma ** 2)) * SPIKE_HEIGHT * 0.8,
                x_range=[center - 3 * sigma, center + 3 * sigma],
                color=BLUE,
                stroke_width=2,
            )

        # new sliding window sprite
        static_bell = bell_path(0)
        static_bell.move_to(rect.get_center())

        # Morph rectangle → bell over the text
        self.play(ReplacementTransform(rect, static_bell), run_time=1.5)

        # slide the slider
        target_pos = self.axes.c2p(rect_width / 2, SPIKE_HEIGHT / 2)
        self.play(static_bell.animate.move_to(target_pos), run_time=1.0)

        def bell_graph():
            c = tracker.get_value()
            bell = bell_path(c)
            bell.set_fill(BLUE, opacity=0.3)
            return bell

        bell_sprite = always_redraw(bell_graph)
        bell_sprite.move_to(target_pos)
        self.add(bell_sprite)
        self.remove(static_bell)

        # raster
        spikes_vmob = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(spikes_vmob))

        # computing curve
        norm = 1 / (math.sqrt(2 * math.pi) * sigma)
        centres = np.arange(0, T, DT)
        weights = np.exp(-((self.time_points[:, None] - centres[None, :]) ** 2) / (2 * sigma ** 2))
        rates = (self.spikes[:, None] * weights).sum(axis=0) * norm
        rate_curve = VMobject(color=RED, stroke_width=2).set_points_as_corners([
            self.axes.c2p(c, r) for c, r in zip(centres, rates)
        ]).stretch(0.5, 1, about_point=self.axes.c2p(0, 0))

        # counter above gaussian sprite
        counter = DecimalNumber(0, num_decimal_places=2)
        counter.add_updater(
            lambda m: m.set_value(rates[min(int(tracker.get_value() / DT), len(rates) - 1)]).next_to(bell_sprite, UP, buff=0.1)
        )
        self.add(counter)

        # play animation
        self.play(tracker.animate.set_value(T - rect_width / 2), Create(rate_curve), run_time=8, rate_func=linear)

        self.play(FadeOut(VGroup(spikes_vmob, bell_sprite, counter, expl1, expl2, gauss_eq)))
        self.wait()




class AlphaHalfWindow(SpikeScene):
    """Causal α‑function smoothing – ignores future spikes."""

    def construct(self):
        self.setup_axes("Rate")
        self.setup_spikes()

        heading = VGroup(
            Text("Gaussian window sees future spikes → biologically implausible.", font_size=24),
            Text("Use a *causal* α‑function instead.", font_size=24),
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(UP)
        self.play(Write(heading[0]))
        self.play(Write(heading[1]))

        alpha_tex = MathTex(r"w(\tau)=[\alpha^{2}\,\tau\,e^{-\alpha\tau}]_{+}").to_corner(UR)
        self.play(FadeIn(alpha_tex, shift=LEFT))

        spikes_vmob = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(spikes_vmob))

        alpha = 15.0
        centres = np.arange(0, T, DT)
        tau = centres[None, :] - self.time_points[:, None]
        weights = np.where(tau >= 0, (alpha ** 2) * tau * np.exp(-alpha * tau), 0)
        rates = (self.spikes[:, None] * weights).sum(axis=0)
        rate_curve = VMobject(color=GREEN, stroke_width=2).set_points_as_corners([
            self.axes.c2p(c, r) for c, r in zip(centres, rates)
        ]).stretch(0.5, 1, about_point=self.axes.c2p(0, 0))

        tracker = ValueTracker(0.0)

        def kernel():
            c = tracker.get_value()
            tau_max = 5 / alpha
            g = self.axes.plot(
                lambda x: (alpha ** 2) * max(x - c, 0) * math.exp(-alpha * max(x - c, 0)) * SPIKE_HEIGHT * 0.8,
                x_range=[c, c + tau_max],
                color=GREEN,
                stroke_width=2,
            )
            g.set_fill(GREEN, opacity=0.3)
            return g

        kernel_sprite = always_redraw(kernel)
        self.add(kernel_sprite)

        counter = DecimalNumber(0, num_decimal_places=2).next_to(kernel_sprite, UP, buff=0.1)
        counter.add_updater(lambda m: m.set_value(rates[min(int(tracker.get_value() / DT), len(rates) - 1)]).next_to(kernel_sprite, UP, buff=0.1))
        self.add(counter)

        self.play(tracker.animate.set_value(T), Create(rate_curve), run_time=8, rate_func=linear)
        self.play(FadeOut(VGroup(spikes_vmob, kernel_sprite, counter, heading, alpha_tex)))
        self.wait()

