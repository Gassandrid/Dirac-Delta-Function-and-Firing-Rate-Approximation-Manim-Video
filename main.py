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
        # Scene 1
        # title
        title = (
            Text("Neurons behave very differently from the binary nature of computers.")
            .scale(0.6)
            .to_edge(UP)
        )
        self.play(Write(title))
        self.wait(0.5)

        # Digital circuit on the left
        bit_axes = create_axes(
            y_range=(0, 2, 1), length=(5, 3), label_x="t", label_y="V (bits)"
        ).to_corner(LEFT + DOWN)

        self.play(Create(bit_axes))

        square = bit_axes.plot(lambda x: 2 if int(x) % 2 == 0 else 0,
                               x_range=[0, 8, 0.01], color=YELLOW)
        self.play(Create(square))
        self.wait(1)  

        # neuron circuit on the right
        neuron_axes = create_axes(
            x_range=(0, 400, 50),
            y_range=(-80, 40, 20),
            length=(5, 3),
            label_x="t (ms)",
            label_y="V (mV)",
        ).to_corner(RIGHT + DOWN)

        self.play(Create(neuron_axes))

        # sim and create the voltage trace
        voltage, dt_ms = get_voltage_trace()
        time_ms = np.arange(len(voltage)) * dt_ms
        points = [neuron_axes.c2p(t, v) for t, v in zip(time_ms, voltage)]
        trace = VMobject(color=YELLOW).set_points_smoothly(points)
        self.play(Create(trace), run_time=4)
        self.wait(1)  

        # equation hint
        lif_eq = (
            MathTex(r"\tau_m \frac{dV}{dt} = -(V-E_L) + \frac{I}{g_L}")
            .scale(0.4)
            .next_to(neuron_axes, UP, buff=0.2)
        )
        self.play(Write(lif_eq))
        self.wait(0.5)
        self.play(FadeOut(lif_eq))

        # Scene 2
        self.play(
            FadeOut(bit_axes),
            FadeOut(square),
            FadeOut(title),
            neuron_axes.animate.move_to(ORIGIN).scale(1.5),
            trace.animate.move_to(ORIGIN).scale(1.5),
            run_time=1.5,
        )

        question = Text("What is the important information here?", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(question))
        self.wait(1)

        spike_times = []
        spike_voltages = []
        threshold = 20  # mV threshold for spike detection
        
        for i, v in enumerate(voltage):
            if v > threshold and (i == 0 or voltage[i-1] <= threshold):
                spike_times.append(time_ms[i])
                spike_voltages.append(v)

        spike_highlights = VGroup()
        for t, v in zip(spike_times, spike_voltages):
            point = neuron_axes.c2p(t, v)
            circle = Circle(radius=0.15, color=RED, fill_opacity=0.8).move_to(point)
            spike_highlights.add(circle)

        self.play(LaggedStart(*[GrowFromCenter(circle) for circle in spike_highlights], lag_ratio=0.3))
        self.wait(1)

        if len(spike_highlights) > 0:
            emphasized_spike = spike_highlights[2] if len(spike_highlights) > 2 else spike_highlights[0]
            self.play(emphasized_spike.animate.scale(1.5).set_color(WHITE), run_time=1)
            self.play(emphasized_spike.animate.scale(1/1.5).set_color(RED), run_time=1)

        self.play(FadeOut(question))
        narration = Text("The fundamental unit of information: action potentials (spikes)", 
                        font_size=28, color=WHITE).to_edge(UP)
        self.play(Write(narration))
        self.wait(1)

        self.play(FadeOut(narration))
        abstraction_title = Text("The Great Abstraction", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(abstraction_title))

        spike_lines = VGroup()
        for t in spike_times:
            line_start = neuron_axes.c2p(t, -80)
            line_end = neuron_axes.c2p(t, 40)
            line = Line(line_start, line_end, color=RED, stroke_width=3)
            spike_lines.add(line)

        self.play(FadeOut(trace), run_time=1)
        self.play(LaggedStart(*[Create(line) for line in spike_lines], lag_ratio=0.1), run_time=2)

        final_text = Text("Discrete events in time", font_size=28, color=WHITE).next_to(abstraction_title, DOWN)
        self.play(Write(final_text))
        self.wait(2)

        self.play(FadeOut(VGroup(abstraction_title, final_text, spike_lines, spike_highlights, neuron_axes)))


class DiracDeltaApproximation(Scene):
    """gossly simplified but visually intuitive Dirac‑delta derivation."""

    def construct(self):
        question = Text("How can we describe something that happens at a single instant?", 
                       font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(question))
        self.wait(1)

        # Create axes
        axes = create_axes(x_range=(-5, 5, 1), y_range=(0, 10, 1), length=(10, 6), 
                          label_x="x", label_y="y").to_edge(DOWN)
        self.play(Create(axes))

        a = ValueTracker(1.0)

        def lorentz(x: float) -> float:
            aa = max(a.get_value(), 0.01)
            return (1 / (math.pi * aa)) * (1 / (1 + (x / aa) ** 2))

        # Create the Lorentzian graph and area
        graph = always_redraw(lambda: axes.plot(lorentz, x_range=[-5, 5, 0.01], color=BLUE))
        area = always_redraw(lambda: axes.get_area(graph, x_range=[-5, 5], color=BLUE_B, opacity=0.3))
        
        label_a = always_redraw(lambda: MathTex(f"a={a.get_value():.2f}").set_color(YELLOW).to_corner(UL))
        
        area_text = MathTex(r"\int_{-\infty}^{\infty} f(x) \, dx = 1", color=GREEN).to_corner(UR)
        
        self.play(Create(graph), Create(area), Write(label_a), Write(area_text))
        self.wait(1)

        self.play(a.animate.set_value(0.05), run_time=5, rate_func=rate_functions.ease_in_cubic)
        self.wait(1)

        narration = Text("Infinitely tall and narrow, but area remains 1", 
                        font_size=24, color=WHITE).next_to(question, DOWN)
        self.play(Write(narration))
        self.wait(1)

        # Transform to idealized delta function
        self.play(FadeOut(question), FadeOut(narration))
        
        # Create the idealized delta function
        vline = Line(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, axes.y_range[1]),
            color=RED, 
            stroke_width=4
        )
        infinity = MathTex(r"\infty", color=RED, font_size=36).next_to(vline, UP)
        delta_label = MathTex(r"\delta(x)", color=RED, font_size=24).next_to(vline, DOWN)

        # Fade out the graph and show the idealized delta
        self.play(
            graph.animate.set_opacity(0.3), 
            area.animate.set_opacity(0.1),
            run_time=1
        )
        self.play(
            Create(vline),
            Write(infinity),
            Write(delta_label),
            run_time=1.5
        )
        
        final_title = Text("The Dirac Delta Function", font_size=36, color=YELLOW).to_edge(UP)
        self.play(Write(final_title))
        self.wait(2)

        self.play(FadeOut(VGroup(axes, graph, area, label_a, area_text, vline, infinity, delta_label, final_title)))


class SpikeTrainAsDeltas(SpikeScene):
    """Shows how to represent a spike train as a sum of delta functions."""

    def construct(self):
        self.setup_axes("Amplitude")
        self.setup_spikes()

        # Scene 4
        title = Text("The Neural Spike Train", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        spike_times = []
        for i, spk in enumerate(self.spikes):
            if spk:
                spike_times.append(self.time_points[i])

        spike_lines = VGroup()
        for t in spike_times:
            line_start = self.axes.c2p(t, 0)
            line_end = self.axes.c2p(t, SPIKE_HEIGHT)
            line = Line(line_start, line_end, color=RED, stroke_width=3)
            spike_lines.add(line)

        self.play(Create(spike_lines))
        self.wait(1)

        delta_symbols = VGroup()
        for t in spike_times:
            point = self.axes.c2p(t, SPIKE_HEIGHT)
            delta = MathTex(r"\delta", color=RED, font_size=24).move_to(point)
            delta_symbols.add(delta)

        self.play(
            FadeOut(spike_lines),
            LaggedStart(*[Write(delta) for delta in delta_symbols], lag_ratio=0.1)
        )
        self.wait(1)

        equation_parts = VGroup()
        
        rho_part = MathTex(r"\rho(t) = ", font_size=36).to_edge(UP).shift(DOWN)
        equation_parts.add(rho_part)
        self.play(Write(rho_part))
        
        sum_part = MathTex(r"\sum_{i} ", font_size=36).next_to(rho_part, RIGHT)
        equation_parts.add(sum_part)
        self.play(Write(sum_part))
        
        delta_part = MathTex(r"\delta(t - t_i)", font_size=36).next_to(sum_part, RIGHT)
        equation_parts.add(delta_part)
        self.play(Write(delta_part))

        full_equation = VGroup(rho_part, sum_part, delta_part)
        self.play(full_equation.animate.to_edge(UP).shift(DOWN * 0.5))
        self.wait(1)

        time_labels = VGroup()
        for i, t in enumerate(spike_times[:5]):
            label = MathTex(f"t_{i}", font_size=20, color=YELLOW)
            label.next_to(self.axes.c2p(t, 0), DOWN, buff=0.1)
            time_labels.add(label)
        
        self.play(LaggedStart(*[Write(label) for label in time_labels], lag_ratio=0.3))
        self.wait(1)

        shift_explanation = Text("Each δ(t - t_i) shifts the delta to time t_i", 
                               font_size=20, color=WHITE).next_to(full_equation, DOWN)
        self.play(Write(shift_explanation))
        self.wait(1)

        sum_explanation = Text("∑ adds up all the deltas", 
                             font_size=20, color=WHITE).next_to(shift_explanation, DOWN)
        self.play(Write(sum_explanation))
        self.wait(1)

        # Use spike_curve animation to give deltas visual height
        collapsed = spike_curve(self.axes, self.time_points, self.spikes, collapsed=True)
        curve = spike_curve(self.axes, self.time_points, self.spikes)
        
        self.play(FadeOut(delta_symbols), FadeOut(time_labels))
        self.play(Create(collapsed))
        self.play(Transform(collapsed, curve, run_time=3))

        final_text = Text("ρ(t): Our fundamental mathematical description of a neuron's output", 
                         font_size=24, color=GREEN).next_to(sum_explanation, DOWN)
        self.play(Write(final_text))
        self.wait(2)

        self.play(FadeOut(VGroup(title, full_equation, shift_explanation, sum_explanation, final_text, collapsed, self.axes)))


class SimpleAverage(SpikeScene):
    """Shows how to calculate the mean firing rate from a spike train."""

    def construct(self):
        self.setup_axes("Amplitude")
        self.setup_spikes()

        # Scene 5
        title = Text("A Simple Average", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        spike_curve_obj = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(spike_curve_obj))
        self.wait(1)

        question = Text("Is this neuron firing 'a lot' or 'a little'?", 
                       font_size=28, color=WHITE).next_to(title, DOWN)
        self.play(Write(question))
        self.wait(1)

        box = Rectangle(
            width=self.axes.x_length,
            height=self.axes.y_length,
            stroke_color=BLUE,
            stroke_width=3,
            fill_opacity=0.1,
            fill_color=BLUE
        ).move_to(self.axes.get_center())
        
        self.play(Create(box))
        self.wait(1)

        total_spikes = int(np.sum(self.spikes))
        spike_count_text = Text(f"Total spikes: {total_spikes}", 
                               font_size=24, color=GREEN).move_to(ORIGIN)
        self.play(Write(spike_count_text))
        self.wait(1)

        calculation = VGroup()
        
        calc_line1 = MathTex(r"\langle r \rangle = \frac{\text{Number of Spikes}}{\text{Time}}", 
                            font_size=28).next_to(question, DOWN, buff=0.5)
        calculation.add(calc_line1)
        self.play(Write(calc_line1))
        
        spike_number = MathTex(str(total_spikes), font_size=24, color=GREEN).move_to(spike_count_text.get_center())
        
        self.play(ReplacementTransform(spike_count_text, spike_number))
        self.play(spike_number.animate.move_to(calc_line1[0][2]).scale(0.8))
        
        time_highlight = Rectangle(
            width=self.axes.x_length,
            height=0.3,
            stroke_color=YELLOW,
            stroke_width=3,
            fill_color=YELLOW,
            fill_opacity=0.3
        ).move_to(self.axes.c2p(5, -0.5))
        
        time_label = Text("Total trial duration", font_size=16, color=YELLOW).next_to(time_highlight, DOWN, buff=0.1)
        
        self.play(Create(time_highlight), Write(time_label))
        self.wait(0.5)
        self.play(FadeOut(time_highlight), FadeOut(time_label))
        
        time_value = MathTex("10\\text{ s}", font_size=28).move_to(calc_line1[0][3])
        self.play(ReplacementTransform(calc_line1[0][3], time_value))
        
        result = MathTex(f"= {total_spikes/10:.1f} \\text{{ Hz}}", font_size=28).next_to(calc_line1, DOWN)
        calculation.add(result)
        self.play(Write(result))
        
        self.wait(1)

        narration = Text("The mean firing rate gives us a single number for the neuron's activity", 
                        font_size=20, color=WHITE).next_to(calculation, DOWN, buff=0.3)
        self.play(Write(narration))
        self.wait(1)

        self.play(FadeOut(narration))
        
        math_title = Text("Averaging Across Trials", font_size=24, color=YELLOW).next_to(calculation, DOWN, buff=0.5)
        self.play(Write(math_title))
        
        trial_title = Text("Generating Multiple Trials", font_size=20, color=WHITE).next_to(math_title, DOWN, buff=0.3)
        self.play(Write(trial_title))
        
        self.play(FadeOut(spike_curve_obj), FadeOut(box))
        
        num_trials = 5
        trial_spikes = []
        trial_counts = []
        current_trial_curve = None
        current_trial_box = None
        
        # Position for trials (same place as original)
        trial_position = self.axes.get_center()
        
        tally_text = Text("Trial counts: ", font_size=18, color=WHITE).next_to(trial_title, DOWN, buff=0.3)
        self.play(Write(tally_text))
        
        for i in range(num_trials):
            _, new_spikes = simulate_spike_train()
            trial_spikes.append(new_spikes)
            
            trial_curve = spike_curve(self.axes, self.time_points, new_spikes)
            trial_curve.move_to(trial_position)
            
            trial_box = Rectangle(
                width=self.axes.x_length,
                height=self.axes.y_length,
                stroke_color=BLUE,
                stroke_width=2,
                fill_opacity=0.1,
                fill_color=BLUE
            ).move_to(trial_position)
            
            trial_count = int(np.sum(new_spikes))
            trial_counts.append(trial_count)
            
            # Fade out previous trial if it exists
            if current_trial_curve is not None:
                self.play(FadeOut(current_trial_curve), FadeOut(current_trial_box))
            
            self.play(Create(trial_curve), Create(trial_box), run_time=0.5)
            current_trial_curve = trial_curve
            current_trial_box = trial_box
            
            tally_update = Text(f"Trial counts: {', '.join(map(str, trial_counts))}", 
                              font_size=18, color=WHITE).next_to(trial_title, DOWN, buff=0.3)
            self.play(ReplacementTransform(tally_text, tally_update))
            tally_text = tally_update
            
            self.wait(0.3)
        
        self.wait(1)
        
        self.play(FadeOut(trial_title), FadeOut(current_trial_curve), FadeOut(current_trial_box))
        avg_title = Text("Averaging the Trials", font_size=20, color=YELLOW).next_to(math_title, DOWN, buff=0.3)
        self.play(Write(avg_title))
        
        avg_spikes = np.mean(trial_counts)
        avg_text = MathTex(f"\\langle n \\rangle = \\frac{{{'+'.join(map(str, trial_counts))}}}{{{num_trials}}} = {avg_spikes:.1f}", 
                          font_size=24).next_to(avg_title, DOWN, buff=0.3)
        self.play(Write(avg_text))
        self.wait(1)
        
        final_avg = MathTex(f"\\langle r \\rangle = \\frac{{{avg_spikes:.1f}}}{{10\\text{{ s}}}} = {avg_spikes/10:.1f} \\text{{ Hz}}", 
                           font_size=24).next_to(avg_text, DOWN, buff=0.3)
        self.play(Write(final_avg))
        self.wait(1)
        
        self.play(FadeOut(VGroup(avg_title, avg_text, final_avg, tally_text)))
        
        eq1 = MathTex(r"\langle r \rangle = \frac{\langle n \rangle}{T}", font_size=28).next_to(math_title, DOWN, buff=0.3)
        self.play(Write(eq1))
        self.wait(1)
        
        eq2 = MathTex(r"\langle n \rangle = \int_0^T d\tau \, \langle \rho(\tau) \rangle", font_size=28).next_to(eq1, DOWN, buff=0.2)
        self.play(Write(eq2))
        self.wait(1)
        
        eq3 = MathTex(r"\langle r \rangle = \frac{1}{T} \int_0^T d\tau \, \langle \rho(\tau) \rangle", font_size=28).next_to(eq2, DOWN, buff=0.2)
        self.play(Write(eq3))
        self.wait(1)
        
        eq4 = MathTex(r"\langle r \rangle = \frac{1}{T} \int_0^T dt \, r(t)", font_size=28).next_to(eq3, DOWN, buff=0.2)
        self.play(Write(eq4))
        self.wait(1)
        
        # Explanation
        explanation = Text("The average firing rate ⟨r⟩ can be computed from the spike train ρ(τ) or firing rate r(t)", 
                          font_size=18, color=WHITE).next_to(eq4, DOWN, buff=0.3)
        self.play(Write(explanation))
        self.wait(2)

        # Clean up for next scene
        self.play(FadeOut(VGroup(title, question, box, spike_count_text, calculation, math_title, eq1, eq2, eq3, eq4, explanation, spike_curve_obj, self.axes)))


class DiscreteBinGraph(SpikeScene):
    """bins the spike train and shows histogram‑like bars."""

    def construct(self):
        self.setup_axes("Count")
        self.setup_spikes()

        # Scene 6
        title = Text("A Rate in Time r(t)", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        spike_curve_obj = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(spike_curve_obj))
        self.wait(1)

        critique = Text("But an average hides the details - spikes aren't evenly spaced!", 
                       font_size=24, color=RED).next_to(title, DOWN)
        self.play(Write(critique))
        self.wait(1)

        rate_intro = Text("What we really want is a rate that changes over time, r(t)", 
                         font_size=24, color=WHITE).next_to(critique, DOWN)
        self.play(Write(rate_intro))
        self.wait(1)

        self.play(FadeOut(critique), FadeOut(rate_intro))

        # ------- discrete bins -------
        bin_lines, containers, bars = create_binned_visualisation(
            self.axes, self.time_points, self.spikes, num_bins=15
        )

        # record for next scene
        Ctx.last_bin_count = 15
        Ctx.last_bin_axes_cfg = dict(x_range=DEFAULT_X_RANGE, y_range=(0, 6, 1))
        Ctx.last_bin_bars = bars

        explanation = Text("Chop time into bins and count spikes in each one", font_size=24).next_to(title, DOWN, buff=0.4)
        self.play(Write(explanation))
        self.play(LaggedStart(*[Create(line) for line in bin_lines], lag_ratio=0.05))
        self.play(LaggedStart(*[Create(cont) for cont in containers], lag_ratio=0.05))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05))
        self.wait(1)

        # demonstrate different bin counts
        problem_text = Text("But the result is blocky and depends on bin size!", font_size=20, color=RED).next_to(explanation, DOWN)
        self.play(Write(problem_text))
        self.wait(1)

        for n in (5, 10, 20, 30):
            nl, nc, nb = create_binned_visualisation(self.axes, self.time_points, self.spikes, n)
            txt = Text(f"Bins: {n}", font_size=24, color=YELLOW).to_corner(RIGHT)
            self.play(
                FadeOut(bin_lines), FadeOut(containers), FadeOut(bars), Write(txt)
            )
            self.play(LaggedStart(*[Create(l) for l in nl], lag_ratio=0.05))
            self.play(LaggedStart(*[Create(c) for c in nc], lag_ratio=0.05))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in nb], lag_ratio=0.05))
            self.wait(1)
            self.play(FadeOut(txt))
            bin_lines, containers, bars = nl, nc, nb

        final_critique = Text("We need a smoother, more principled approach", font_size=24, color=GREEN).next_to(problem_text, DOWN)
        self.play(Write(final_critique))
        self.wait(2)

        self.play(FadeOut(VGroup(title, explanation, problem_text, final_critique, spike_curve_obj, bin_lines, containers, bars, self.axes)))


class SlidingWindowApprox(SpikeScene):
    """Starts with bars from previous scene, fades them, then sliding rectangular window."""

    def construct(self):
        self.setup_axes("Count")
        self.setup_spikes()

        title = Text("The Sliding Window", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        trace = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(trace))
        self.wait(1)

        intro = Text("Instead of fixed bins, use a sliding window", font_size=24, color=WHITE).next_to(title, DOWN)
        self.play(Write(intro))
        self.wait(1)

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

        self.play(tracker.animate.set_value(T - half), Create(count_curve), run_time=6, rate_func=linear)
        
        critique = Text("But rectangular windows cause sudden jumps!", font_size=20, color=RED).next_to(intro, DOWN)
        self.play(Write(critique))
        self.wait(1)

        spike_indices = np.where(self.spikes)[0]
        if len(spike_indices) > 0:
            mid_spike_time = self.time_points[spike_indices[len(spike_indices)//2]]
            
            spike_point = self.axes.c2p(mid_spike_time, SPIKE_HEIGHT)
            spike_highlight = Circle(radius=0.1, color=YELLOW, fill_opacity=0.8).move_to(spike_point)
            self.play(GrowFromCenter(spike_highlight))
            
            self.play(tracker.animate.set_value(mid_spike_time - 0.3), run_time=1)
            self.play(tracker.animate.set_value(mid_spike_time + 0.3), run_time=1)
            
            self.play(FadeOut(spike_highlight))

        self.play(FadeOut(VGroup(trace, win, counter, intro, critique)))
        self.wait(1)


class GaussianWindow(SpikeScene):
    """Smooths the rectangular sliding‑window estimate with a Gaussian kernel."""

    def construct(self):
        self.setup_axes("Count")
        self.setup_spikes()

        # Scene 7 continued
        title = Text("A Better Window: Gaussian", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        # why we need a gaussian
        expl1 = Text(
            "A much better window weighs spikes at the center more than at the edges",
            font_size=24
        ).next_to(title, DOWN)
        self.play(Write(expl1))

        rect_width = 0.5
        rect = Rectangle(
            width=self.axes.x_axis.unit_size * rect_width,
            height=self.axes.y_length,
            stroke_color=GREY,
            stroke_width=2,
            fill_color=BLUE,
            fill_opacity=0.5,
        )
        rect.move_to(self.axes.c2p(2, SPIKE_HEIGHT / 2))
        self.play(Create(rect))

        # gaussian helper
        sigma = 0.25
        tracker = ValueTracker(2.0)  # will be reset later

        def bell_path(center: float):
            return self.axes.plot(
                lambda x: math.exp(-((x - center) ** 2) / (2 * sigma ** 2)) * SPIKE_HEIGHT * 0.8,
                x_range=[center - 3 * sigma, center + 3 * sigma],
                color=BLUE,
                stroke_width=2,
            )

        # new sliding window sprite
        static_bell = bell_path(2.0)
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

        self.play(tracker.animate.set_value(T - rect_width / 2), Create(rate_curve), run_time=6, rate_func=linear)

        self.play(FadeOut(expl1))
        conv_eq = MathTex(r"r(t) = w(t) * \rho(t)", font_size=36, color=YELLOW).to_edge(UP).shift(DOWN)
        self.play(Write(conv_eq))
        
        conv_explanation = Text("Sliding the window w over the spike train ρ and calculating the weighted sum", 
                              font_size=20, color=WHITE).next_to(conv_eq, DOWN)
        self.play(Write(conv_explanation))
        self.wait(2)

        self.play(FadeOut(VGroup(spikes_vmob, bell_sprite, counter, conv_eq, conv_explanation, self.axes)))
        self.wait(1)


class AlphaHalfWindow(SpikeScene):
    """Causal α‑function smoothing – ignores future spikes."""

    def construct(self):
        self.setup_axes("Rate")
        self.setup_spikes()

        # Scene 8
        title = Text("A Causal Window", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))

        sigma = 0.25
        centres = np.arange(0, T, DT)
        weights = np.exp(-((self.time_points[:, None] - centres[None, :]) ** 2) / (2 * sigma ** 2))
        norm = 1 / (math.sqrt(2 * math.pi) * sigma)
        rates = (self.spikes[:, None] * weights).sum(axis=0) * norm
        gaussian_curve = VMobject(color=BLUE, stroke_width=2).set_points_as_corners([
            self.axes.c2p(c, r) for c, r in zip(centres, rates)
        ]).stretch(0.5, 1, about_point=self.axes.c2p(0, 0))

        self.play(Create(gaussian_curve))
        self.wait(1)

        problem = Text("Biological problem: Gaussian window 'sees' into the future!", 
                      font_size=24, color=RED).next_to(title, DOWN)
        self.play(Write(problem))
        self.wait(1)

        # Freeze the Gaussian window over a spike to show symmetry
        spike_indices = np.where(self.spikes)[0]
        if len(spike_indices) > 0:
            mid_spike_time = self.time_points[spike_indices[len(spike_indices)//2]]
            
            # Create a symmetric Gaussian window
            def gaussian_window(center: float):
                return self.axes.plot(
                    lambda x: math.exp(-((x - center) ** 2) / (2 * sigma ** 2)) * SPIKE_HEIGHT * 0.8,
                    x_range=[center - 3 * sigma, center + 3 * sigma],
                    color=BLUE,
                    stroke_width=2,
                )
            
            frozen_window = gaussian_window(mid_spike_time)
            frozen_window.set_fill(BLUE, opacity=0.3)
            frozen_window.move_to(self.axes.c2p(mid_spike_time, SPIKE_HEIGHT / 2))
            
            self.play(Create(frozen_window))
            self.wait(1)
            
            # Show the symmetry problem
            symmetry_text = Text("Symmetric window sees past AND future", 
                               font_size=20, color=RED).next_to(problem, DOWN)
            self.play(Write(symmetry_text))
            self.wait(1)
            
            self.play(FadeOut(frozen_window), FadeOut(symmetry_text))

        # Introduce causal windows
        causal_intro = Text("Neuroscientists use causal windows that only consider the past", 
                           font_size=24, color=GREEN).next_to(problem, DOWN)
        self.play(Write(causal_intro))
        self.wait(1)

        # Fade out Gaussian curve and show spike train
        self.play(FadeOut(gaussian_curve))
        spikes_vmob = spike_curve(self.axes, self.time_points, self.spikes)
        self.play(Create(spikes_vmob))

        # Show alpha function
        alpha_tex = MathTex(r"w(\tau)=[\alpha^{2}\,\tau\,e^{-\alpha\tau}]_{+}", font_size=28).to_corner(UR)
        self.play(FadeIn(alpha_tex, shift=LEFT))

        alpha = 15.0
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

        self.play(tracker.animate.set_value(T), Create(rate_curve), run_time=6, rate_func=linear)
        
        final_text = Text("Causal rate rises after bursts - biologically plausible!", 
                         font_size=20, color=GREEN).next_to(causal_intro, DOWN)
        self.play(Write(final_text))
        self.wait(2)

        self.play(FadeOut(VGroup(spikes_vmob, kernel_sprite, counter, problem, causal_intro, final_text, alpha_tex, self.axes)))


class TuningCurve(Scene):
    """Demonstrates how firing rates encode stimulus information through tuning curves."""

    def construct(self):
        title = Text("The Neural Code: What Does the Rate Mean?", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        stimulus_title = Text("Stimulus", font_size=24, color=WHITE).to_corner(UL).shift(DOWN)
        self.play(Write(stimulus_title))

        bar_length = 2
        bar = Line(start=[-bar_length/2, 0, 0], end=[bar_length/2, 0, 0], 
                  stroke_width=8, color=WHITE)
        bar.move_to(LEFT * 3 + DOWN)
        bar_current_angle = 0  # Track the current angle manually
        self.play(Create(bar))

        dial_radius = 0.8
        dial = Circle(radius=dial_radius, stroke_color=WHITE, stroke_width=2)
        dial.move_to(bar.get_center() + UP * 1.5)
        
        angle_line = Line(dial.get_center(), dial.get_center() + RIGHT * dial_radius, 
                         stroke_color=RED, stroke_width=3)
        angle_line_current_angle = 0  # Track the current angle manually
        
        angle_label = MathTex(r"\text{Angle} = 0°", font_size=20, color=RED)
        angle_label.next_to(dial, UP)
        
        self.play(Create(dial), Create(angle_line), Write(angle_label))
        self.wait(1)

        response_title = Text("Neural Response", font_size=24, color=WHITE).to_corner(UR).shift(DOWN)
        self.play(Write(response_title))

        spike_axes = create_axes(
            x_range=(0, 2, 0.5), y_range=(0, 1, 0.2), 
            length=(3, 2), label_x="Time (s)", label_y=""
        ).move_to(RIGHT * 3 + DOWN)
        self.play(Create(spike_axes))

        tuning_axes = create_axes(
            x_range=(0, 180, 30), y_range=(0, 50, 10), 
            length=(6, 3), label_x="Stimulus Orientation (degrees)", label_y="Firing Rate (Hz)"
        ).move_to(DOWN * 2.5)
        self.play(Create(tuning_axes))

        def generate_spike_train_for_orientation(angle_deg):
            max_rate = 45  # Hz
            min_rate = 5   # Hz
            preferred_angle = 90  # degrees
            
            rate = min_rate + (max_rate - min_rate) * np.exp(-((angle_deg - preferred_angle) ** 2) / (2 * 30 ** 2))
            
            dt = 0.01
            t = np.arange(0, 2, dt)
            spikes = np.random.rand(len(t)) < rate * dt
            return t, spikes, rate

        angles = [0, 30, 45, 60, 90, 120, 135, 150, 180]
        measured_rates = []
        tuning_points = []

        for angle in angles:
            new_angle_rad = np.radians(angle)
            angle_diff = new_angle_rad - bar_current_angle
            bar.rotate(angle_diff)
            bar_current_angle = new_angle_rad
            
            angle_diff_line = new_angle_rad - angle_line_current_angle
            angle_line.rotate(angle_diff_line)
            angle_line_current_angle = new_angle_rad
            
            new_angle_label = MathTex(f"\\text{Angle} = {angle}°", font_size=20, color=RED)
            new_angle_label.next_to(dial, UP)
            self.play(
                Transform(angle_label, new_angle_label),
                run_time=0.5
            )

            t, spikes, rate = generate_spike_train_for_orientation(angle)
            spike_curve_obj = spike_curve(spike_axes, t, spikes, height=0.8)
            
            # clear previous spike train
            if hasattr(self, 'current_spike_curve'):
                self.play(FadeOut(self.current_spike_curve))
            
            self.play(Create(spike_curve_obj), run_time=1)
            self.current_spike_curve = spike_curve_obj
            
            rate_text = Text(f"Rate: {rate:.1f} Hz", font_size=18, color=GREEN).next_to(spike_axes, UP)
            if hasattr(self, 'current_rate_text'):
                self.play(ReplacementTransform(self.current_rate_text, rate_text))
            else:
                self.play(Write(rate_text))
            self.current_rate_text = rate_text
            
            measured_rates.append(rate)
            
            # plot
            point = Dot(tuning_axes.c2p(angle, rate), color=RED, radius=0.08)
            tuning_points.append(point)
            self.play(Create(point))
            
            self.wait(0.5)

        curve_points = [tuning_axes.c2p(angle, rate) for angle, rate in zip(angles, measured_rates)]
        tuning_curve = VMobject(color=BLUE, stroke_width=3).set_points_smoothly(curve_points)
        self.play(Create(tuning_curve))
        
        explanation = Text("This is the neuron's tuning curve - its preference map", 
                          font_size=20, color=BLUE).next_to(tuning_axes, DOWN)
        self.play(Write(explanation))
        self.wait(2)

        self.play(FadeOut(VGroup(title, stimulus_title, response_title, bar, dial, angle_line, angle_label, 
                                spike_axes, tuning_axes, tuning_curve, explanation, self.current_spike_curve, 
                                self.current_rate_text, *tuning_points)))


class DecodingMind(Scene):
    """Demonstrates how to decode stimulus information from firing rates using tuning curves."""

    def construct(self):
        title = Text("Decoding the Mind", font_size=32, color=YELLOW).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        tuning_axes = create_axes(
            x_range=(0, 180, 30), y_range=(0, 50, 10), 
            length=(6, 3), label_x="Stimulus Orientation (degrees)", label_y="Firing Rate (Hz)"
        ).move_to(LEFT * 2)
        self.play(Create(tuning_axes))

        angles = np.linspace(0, 180, 100)
        rates = 5 + 40 * np.exp(-((angles - 90) ** 2) / (2 * 30 ** 2))
        curve_points = [tuning_axes.c2p(angle, rate) for angle, rate in zip(angles, rates)]
        tuning_curve = VMobject(color=BLUE, stroke_width=3).set_points_smoothly(curve_points)
        self.play(Create(tuning_curve))

        stimulus_title = Text("Stimulus (Hidden)", font_size=24, color=WHITE).to_corner(UR).shift(DOWN)
        self.play(Write(stimulus_title))

        bar = Line(start=[-1, 0, 0], end=[1, 0, 0], stroke_width=8, color=WHITE)
        bar.move_to(RIGHT * 3 + DOWN)
        question_mark = Text("?", font_size=48, color=YELLOW).next_to(bar, UP)
        self.play(Create(bar), Write(question_mark))

        spike_axes = create_axes(
            x_range=(0, 2, 0.5), y_range=(0, 1, 0.2), 
            length=(3, 2), label_x="Time (s)", label_y=""
        ).move_to(RIGHT * 3 + UP)
        self.play(Create(spike_axes))

        target_rate = 35
        dt = 0.01
        t = np.arange(0, 2, dt)
        spikes = np.random.rand(len(t)) < target_rate * dt
        spike_curve_obj = spike_curve(spike_axes, t, spikes, height=0.8)
        self.play(Create(spike_curve_obj), run_time=2)

        rate_text = Text(f"Measured Rate: {target_rate} Hz", font_size=18, color=GREEN).next_to(spike_axes, UP)
        self.play(Write(rate_text))

        decoding_title = Text("Decoding Process", font_size=20, color=YELLOW).next_to(title, DOWN)
        self.play(Write(decoding_title))

        # Horizontal line to the tuning curve
        horizontal_line = Line(
            start=tuning_axes.c2p(0, target_rate),
            end=tuning_axes.c2p(180, target_rate),
            color=RED, stroke_width=2
        )
        self.play(Create(horizontal_line))

        # Vertical line down to x-axis
        target_angle = angles[np.argmin(np.abs(rates - target_rate))]
        vertical_line = Line(
            start=tuning_axes.c2p(target_angle, target_rate),
            end=tuning_axes.c2p(target_angle, 0),
            color=RED, stroke_width=2
        )
        self.play(Create(vertical_line))

        # Show decoded angle
        decoded_angle = MathTex(f"\\text{Angle} = {target_angle:.0f}°", font_size=24, color=RED)
        decoded_angle.next_to(tuning_axes, DOWN)
        self.play(Write(decoded_angle))

        # Rotate the bar to show the decoded orientation
        new_angle_rad = np.radians(target_angle)
        bar.rotate(new_angle_rad)
        self.play(
            FadeOut(question_mark),
            run_time=1
        )

        final_explanation = Text("By listening to spikes, we can read the mind!", 
                               font_size=20, color=GREEN).next_to(decoded_angle, DOWN)
        self.play(Write(final_explanation))
        self.wait(2)

        self.play(FadeOut(VGroup(title, decoding_title, tuning_axes, tuning_curve, stimulus_title, 
                                bar, spike_axes, spike_curve_obj, rate_text, horizontal_line, 
                                vertical_line, decoded_angle, final_explanation)))

