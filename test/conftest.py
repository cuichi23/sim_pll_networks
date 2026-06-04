"""Shared pytest configuration for the sim_pll test suite."""
import matplotlib

matplotlib.use("Agg")  # headless backend; setup_topology() draws the graph as a side effect
