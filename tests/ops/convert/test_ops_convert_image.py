import numpy as np
import pytest

from raygeo.image.scan import ScanMode, generate_scan_lines
from raygeo.ops import Ops
from raygeo.ops.types import CommandType


def _scan_segments(ops):
    return [
        (ops.endpoint(index - 1), ops.endpoint(index))
        for index in range(ops.len())
        if ops.command_type(index) == CommandType.SCAN_LINE
    ]


class TestFromPowerModulatedImage:
    def test_empty_alpha(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        alpha = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray, alpha, (10.0, 10.0), 0.0, 0.0, 0.1, 0.05
        )
        assert ops.is_empty()

    def test_full_image(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray, alpha, (10.0, 10.0), 0.0, 0.0, 0.1, 0.05
        )
        assert not ops.is_empty()

    def test_white_image_empty(self):
        gray = np.full((10, 10), 255, dtype=np.uint8)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            0.05,
            min_power=0.0,
            max_power=1.0,
        )
        assert ops.is_empty()

    def test_with_power_quantization(self):
        gray = np.full((10, 10), 64, dtype=np.uint8)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            0.05,
            num_power_levels=4,
        )
        assert not ops.is_empty()

    def test_power_quantization_preserves_scaled_output_bounds(self):
        gray = np.array([[255, 128, 0]], dtype=np.uint8)
        alpha = np.full_like(gray, 255)
        ops = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            0.05,
            min_power=0.0,
            max_power=1.0,
            step_power=0.2,
            num_power_levels=25,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        samples = [
            value
            for index in ops.indices_of(CommandType.SCAN_LINE)
            for value in ops.scanline_data(index)
        ]
        assert samples
        assert max(samples) == round(0.2 * 255)

    def test_with_angle(self):
        gray = np.full((20, 20), 128, dtype=np.uint8)
        alpha = np.full((20, 20), 255, dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray, alpha, (10.0, 10.0), 0.0, 0.0, 0.1, 0.05, angle=45.0
        )
        assert not ops.is_empty()


class TestFromMaskScan:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
        assert ops.is_empty()

    def test_full_mask(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
        assert not ops.is_empty()

    def test_unidirectional_marks_share_direction_with_dark_returns(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_scan(
            mask,
            (10.0, 10.0),
            0.0,
            0.0,
            0.2,
            bidirectional=False,
        )
        scans = ops.indices_of(CommandType.SCAN_LINE)

        assert len(scans) > 2
        segments = [
            (ops.endpoint(index - 1), ops.endpoint(index)) for index in scans
        ]
        assert all(end[0] > start[0] for start, end in segments)
        for previous, current in zip(scans, scans[1:]):
            move = current - 1
            assert ops.command_type(move) == CommandType.MOVE_TO
            assert ops.endpoint(previous)[0] > ops.endpoint(move)[0]

    def test_explicit_bidirectional_preserves_default_output(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        default = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.2)
        explicit = Ops.from_mask_scan(
            mask,
            (10.0, 10.0),
            0.0,
            0.0,
            0.2,
            bidirectional=True,
        )

        assert explicit.to_dict() == default.to_dict()

    def test_step_power(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.1, step_power=0.5
        )
        assert not ops.is_empty()

    def test_with_angle(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        ops = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.1, angle=90.0)
        assert not ops.is_empty()

    @pytest.mark.parametrize(
        "scan_mode", [ScanMode.SEGMENTED, ScanMode.FULL_SWEEP]
    )
    def test_anisotropic_endpoint_extension_stays_parallel(self, scan_mode):
        height, width = 180, 24
        pixels_per_mm = (2.0, 10.0)
        line_interval_mm = 4.0
        mask = np.ones((height, width), dtype=np.uint8)
        angle = 33.0
        ops = Ops.from_mask_scan(
            mask,
            pixels_per_mm,
            0.0,
            0.0,
            line_interval_mm,
            angle=angle,
            scan_mode=scan_mode,
        )
        scan_lines = generate_scan_lines(
            (0, height - 1, 0, width - 1),
            (width, height),
            pixels_per_mm,
            line_interval_mm,
            angle,
        )
        angle_rad = np.deg2rad(angle)
        direction = np.array([np.cos(angle_rad), -np.sin(angle_rad)])
        segments = _scan_segments(ops)

        assert segments
        assert len(scan_lines) == len(segments)
        ymax_mm = height / pixels_per_mm[1]
        for scan_line, (start, end) in zip(scan_lines, segments):
            delta = np.subtract(end[:2], start[:2])
            cross = delta[0] * direction[1] - delta[1] * direction[0]
            assert abs(cross) < 1e-12

            first, last = scan_line.pixels[0], scan_line.pixels[-1]
            if scan_line.index % 2:
                first, last = last, first
            center_start = scan_line.pixel_to_mm(*first, pixels_per_mm)
            center_end = scan_line.pixel_to_mm(*last, pixels_per_mm)
            center_start = (center_start[0], ymax_mm - center_start[1])
            center_end = (center_end[0], ymax_mm - center_end[1])
            center_length = np.linalg.norm(
                np.subtract(center_end, center_start)
            )
            extended_length = np.linalg.norm(delta)
            scan_direction = np.array(scan_line.direction())
            directional_density = np.linalg.norm(
                scan_direction * np.array(pixels_per_mm)
            )
            np.testing.assert_allclose(
                extended_length - center_length,
                1.0 / directional_density,
                rtol=0,
                atol=1e-12,
            )

    def test_isotropic_diagonal_endpoint_extension_is_unchanged(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        ops = Ops.from_mask_scan(
            mask,
            (4.0, 4.0),
            0.0,
            0.0,
            2.0,
            angle=45.0,
        )

        np.testing.assert_allclose(
            _scan_segments(ops),
            [
                (
                    (0.03661165235168168, 1.9633883476483183, 0.0),
                    (1.963388347648318, 0.036611652351681734, 0.0),
                )
            ],
            rtol=0,
            atol=1e-12,
        )


class TestFromMaskLines:
    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_lines(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
        assert ops.is_empty()

    def test_full_mask(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_lines(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
        assert not ops.is_empty()

    def test_with_z_offset(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_lines(mask, (10.0, 10.0), 0.0, 0.0, 0.1, z=-2.0)
        assert not ops.is_empty()


class TestFromMultiPassImage:
    def test_white_image_empty(self):
        gray = np.full((10, 10), 255, dtype=np.uint8)
        ops = Ops.from_multi_pass_image(
            gray, (10.0, 10.0), 0.0, 0.0, 0.1, 5, 0.5
        )
        assert ops.is_empty()

    def test_dark_image(self):
        gray = np.full((10, 10), 0, dtype=np.uint8)
        ops = Ops.from_multi_pass_image(
            gray, (10.0, 10.0), 0.0, 0.0, 0.1, 3, 0.5
        )
        assert not ops.is_empty()

    def test_gradient(self):
        gray = np.zeros((20, 20), dtype=np.uint8)
        for i in range(20):
            gray[i, :] = int(i * 255 / 19)
        ops = Ops.from_multi_pass_image(
            gray, (10.0, 10.0), 0.0, 0.0, 0.1, 3, 0.5
        )
        assert not ops.is_empty()

    def test_with_angle_increment(self):
        gray = np.full((10, 10), 64, dtype=np.uint8)
        ops = Ops.from_multi_pass_image(
            gray,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            3,
            0.5,
            angle=0.0,
            angle_increment=45.0,
        )
        assert not ops.is_empty()


class TestFullSweepPowerModulation:
    def _make_images(self, size=30, gray_val=64):
        gray = np.full((size, size), gray_val, dtype=np.uint8)
        alpha = np.full((size, size), 255, dtype=np.uint8)
        return gray, alpha

    def test_fewer_scans_than_segmented(self):
        gray, alpha = self._make_images()
        seg = Ops.from_power_modulated_image(
            gray, alpha, (10.0, 10.0), 0, 0, 0.1, 0.05
        )
        fs = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0,
            0,
            0.1,
            0.05,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        assert fs.len() <= seg.len()

    def test_produces_scan_lines(self):
        gray, alpha = self._make_images()
        ops = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0,
            0,
            0.1,
            0.05,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        types = [ops.command_type(i) for i in range(ops.len())]
        assert CommandType.SCAN_LINE in types

    def test_empty_alpha(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        alpha = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0,
            0,
            0.1,
            0.05,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        assert ops.is_empty()


class TestFullSweepMaskScan:
    def test_fewer_scans_than_segmented(self):
        mask = np.ones((30, 30), dtype=np.uint8)
        seg = Ops.from_mask_scan(mask, (10.0, 10.0), 0, 0, 0.1)
        fs = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        assert fs.len() <= seg.len()

    def test_produces_scan_lines(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        ops = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        types = [ops.command_type(i) for i in range(ops.len())]
        assert CommandType.SCAN_LINE in types

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        assert ops.is_empty()


class TestFullSweepMaskLines:
    def test_fewer_commands_than_segmented(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:15] = 1
        mask[5:25, 18:25] = 1
        seg = Ops.from_mask_lines(mask, (10.0, 10.0), 0, 0, 0.1)
        fs = Ops.from_mask_lines(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        assert fs.len() < seg.len()

    def test_produces_scanlines(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        ops = Ops.from_mask_lines(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        types = [ops.command_type(i) for i in range(ops.len())]
        assert CommandType.SCAN_LINE in types

    def test_empty_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        ops = Ops.from_mask_lines(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        assert ops.is_empty()

    def test_power_zero_in_gaps(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:25] = 1
        ops = Ops.from_mask_lines(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        scan_indices = [
            i
            for i in range(ops.len())
            if ops.command_type(i) == CommandType.SCAN_LINE
        ]
        assert len(scan_indices) > 0
        for i in scan_indices:
            pv = ops.scanline_data(i)
            assert min(pv) == 0
            assert max(pv) > 0

    def test_one_scanline_per_scan_row(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:25, 5:15] = 1
        mask[5:25, 18:25] = 1
        ops = Ops.from_mask_lines(
            mask, (10.0, 10.0), 0, 0, 0.1, scan_mode=ScanMode.FULL_SWEEP
        )
        scan_count = sum(
            1
            for i in range(ops.len())
            if ops.command_type(i) == CommandType.SCAN_LINE
        )
        seg = Ops.from_mask_lines(mask, (10.0, 10.0), 0, 0, 0.1)
        line_count = sum(
            1
            for i in range(seg.len())
            if seg.command_type(i) == CommandType.LINE_TO
        )
        assert scan_count < line_count


class TestFullSweepMultiPass:
    def test_fewer_commands_than_segmented(self):
        gray = np.full((20, 20), 255, dtype=np.uint8)
        gray[5:15, 5:10] = 64
        gray[5:15, 12:15] = 64
        seg = Ops.from_multi_pass_image(gray, (10.0, 10.0), 0, 0, 0.1, 3, 0.5)
        fs = Ops.from_multi_pass_image(
            gray,
            (10.0, 10.0),
            0,
            0,
            0.1,
            3,
            0.5,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        assert fs.len() < seg.len()

    def test_produces_scanlines(self):
        gray = np.full((20, 20), 64, dtype=np.uint8)
        ops = Ops.from_multi_pass_image(
            gray,
            (10.0, 10.0),
            0,
            0,
            0.1,
            3,
            0.5,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        types = [ops.command_type(i) for i in range(ops.len())]
        assert CommandType.SCAN_LINE in types

    def test_white_image_empty(self):
        gray = np.full((10, 10), 255, dtype=np.uint8)
        ops = Ops.from_multi_pass_image(
            gray,
            (10.0, 10.0),
            0,
            0,
            0.1,
            3,
            0.5,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        assert ops.is_empty()


def test_from_mask_lines():
    mask = np.ones((10, 10), dtype=np.uint8)
    ops = Ops.from_mask_lines(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
    assert not ops.is_empty()


def test_from_mask_lines_empty():
    mask = np.zeros((10, 10), dtype=np.uint8)
    ops = Ops.from_mask_lines(mask, (10.0, 10.0), 0.0, 0.0, 0.1)
    assert ops.is_empty()


def test_scan_mode_enum():
    assert ScanMode.SEGMENTED is not None
    assert ScanMode.FULL_SWEEP is not None


class TestDotWidthCorrection:
    """Must never change toolpath geometry, only which power samples fire."""

    def _endpoints(self, ops):
        return [ops.endpoint(i) for i in range(ops.len())]

    def _command_types(self, ops):
        return [ops.command_type(i) for i in range(ops.len())]

    def test_mask_scan_geometry_unchanged_segmented(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        baseline = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0)
        trimmed = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0, dot_width_correction_mm=0.2
        )
        assert self._command_types(baseline) == self._command_types(trimmed)
        for a, b in zip(self._endpoints(baseline), self._endpoints(trimmed)):
            assert a == b

    def test_mask_scan_geometry_unchanged_full_sweep(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        baseline = Ops.from_mask_scan(
            mask,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            1.0,
            scan_mode=ScanMode.FULL_SWEEP,
        )
        trimmed = Ops.from_mask_scan(
            mask,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            1.0,
            scan_mode=ScanMode.FULL_SWEEP,
            dot_width_correction_mm=0.2,
        )
        assert self._command_types(baseline) == self._command_types(trimmed)
        for a, b in zip(self._endpoints(baseline), self._endpoints(trimmed)):
            assert a == b

    def test_mask_scan_trims_power_at_each_end(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        # 10 px/mm, 0.2mm correction -> trim 2 samples off each end.
        trimmed = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0, dot_width_correction_mm=0.2
        )
        data = trimmed.scanline_data(1)
        assert list(data[:2]) == [0, 0]
        assert list(data[-2:]) == [0, 0]
        assert all(v > 0 for v in data[2:-2])

    def test_power_modulated_geometry_unchanged(self):
        gray = np.full((10, 10), 0, dtype=np.uint8)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        baseline = Ops.from_power_modulated_image(
            gray, alpha, (10.0, 10.0), 0.0, 0.0, 0.1, 0.02
        )
        trimmed = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            0.02,
            dot_width_correction_mm=0.2,
        )
        assert self._command_types(baseline) == self._command_types(trimmed)
        for a, b in zip(self._endpoints(baseline), self._endpoints(trimmed)):
            assert a == b

    def test_power_modulated_trims_power_at_each_end(self):
        gray = np.full((10, 10), 0, dtype=np.uint8)
        alpha = np.full((10, 10), 255, dtype=np.uint8)
        trimmed = Ops.from_power_modulated_image(
            gray,
            alpha,
            (10.0, 10.0),
            0.0,
            0.0,
            0.1,
            0.02,
            dot_width_correction_mm=0.2,
        )
        data = trimmed.scanline_data(1)
        assert list(data[:2]) == [0, 0]
        assert list(data[-2:]) == [0, 0]

    def test_zero_correction_is_no_op(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        baseline = Ops.from_mask_scan(mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0)
        explicit_zero = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0, dot_width_correction_mm=0.0
        )
        assert baseline.scanline_data(1) == explicit_zero.scanline_data(1)

    def test_trim_larger_than_segment_zeroes_whole_run(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        trimmed = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.1, 1.0, dot_width_correction_mm=5.0
        )
        data = trimmed.scanline_data(1)
        assert all(v == 0 for v in data)

    def test_geometry_unchanged_at_nonzero_angle(self):
        mask = np.ones((20, 20), dtype=np.uint8)
        baseline = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.2, 1.0, angle=30.0
        )
        trimmed = Ops.from_mask_scan(
            mask,
            (10.0, 10.0),
            0.0,
            0.0,
            0.2,
            1.0,
            angle=30.0,
            dot_width_correction_mm=0.2,
        )
        assert self._command_types(baseline) == self._command_types(trimmed)
        for a, b in zip(self._endpoints(baseline), self._endpoints(trimmed)):
            assert a == b

        scan_indices = [
            i
            for i, ct in enumerate(self._command_types(trimmed))
            if ct == CommandType.SCAN_LINE
        ]
        assert scan_indices
        # Pick the longest line; corner ones can be fully trimmed away.
        data = max((trimmed.scanline_data(i) for i in scan_indices), key=len)
        assert len(data) > 4
        assert data[0] == 0
        assert data[-1] == 0
        assert any(v > 0 for v in data)

    def test_multiple_segments_trimmed_independently(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[:, 0:8] = 1
        mask[:, 12:20] = 1

        trimmed = Ops.from_mask_scan(
            mask, (10.0, 10.0), 0.0, 0.0, 0.2, 1.0, dot_width_correction_mm=0.1
        )

        scan_indices = [
            i
            for i in range(trimmed.len())
            if trimmed.command_type(i) == CommandType.SCAN_LINE
        ]
        assert len(scan_indices) >= 2

        first_seg = trimmed.scanline_data(scan_indices[0])
        second_seg = trimmed.scanline_data(scan_indices[1])
        for seg in (first_seg, second_seg):
            assert seg[0] == 0
            assert seg[-1] == 0
            assert any(v > 0 for v in seg)
