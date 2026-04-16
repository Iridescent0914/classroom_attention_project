#!/usr/bin/env python3
"""Compatibility entrypoint for LLM classroom report generation."""

from classroom_attention.reporting import run_llm_reporter


if __name__ == "__main__":
    run_llm_reporter()