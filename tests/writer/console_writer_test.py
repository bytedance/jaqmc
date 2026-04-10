# Copyright (c) 2025-2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from jaqmc.writer.console import ConsoleWriter


def test_console_writer_default(mocker):
    writer = ConsoleWriter(interval=1)
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        stats = {"loss": 1.234567, "other": 9.99}
        writer.write(1, stats)

        # Expect step and loss (default field)
        mock_logger.info.assert_called_with("step=1, loss=1.2346")


def test_console_writer_custom_fields(mocker):
    writer = ConsoleWriter(interval=1, fields="loss, other")
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        stats = {"loss": 1.234567, "other": 9.99}
        writer.write(1, stats)

        mock_logger.info.assert_called_with("step=1, loss=1.2346, other=9.9900")


def test_console_writer_custom_format_spec(mocker):
    writer = ConsoleWriter(interval=1, fields="loss:.2f, other:.1f")
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        stats = {"loss": 1.234567, "other": 9.99}
        writer.write(1, stats)

        mock_logger.info.assert_called_with("step=1, loss=1.23, other=10.0")


def test_console_writer_alias(mocker):
    writer = ConsoleWriter(
        interval=1, fields="E=total_energy, Lz=angular_momentum_z:+.4f"
    )
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        stats = {"total_energy": 2.6933, "angular_momentum_z": -0.1255}
        writer.write(1, stats)

        mock_logger.info.assert_called_with("step=1, E=2.6933, Lz=-0.1255")


def test_console_writer_colon_in_key(mocker):
    writer = ConsoleWriter(
        interval=1,
        fields="energy:kinetic, kinetic=energy:kinetic:.4f, energy:potential",
    )
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        stats = {"energy:kinetic": 1.5, "energy:potential": -0.75}
        writer.write(1, stats)

        mock_logger.info.assert_called_with(
            "step=1, energy:kinetic=1.5000, kinetic=1.5000, energy:potential=-0.7500"
        )


def test_console_writer_interval(mocker):
    writer = ConsoleWriter(interval=10)
    with writer.open(None, "test"):
        mock_logger = mocker.patch.object(writer, "logger")
        writer.write(5, {"loss": 1.0})
        mock_logger.info.assert_not_called()

        writer.write(10, {"loss": 1.0})
        mock_logger.info.assert_called()
