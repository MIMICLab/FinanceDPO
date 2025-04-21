[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "dpo-forecasting"
version = "0.1.0"
description = "Direct Preference Optimization for decision-driven financial time series forecasting"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
  {name = "Your Name", email = "you@example.com"}
]
readme = "README.md"
classifiers = [
  "Programming Language :: Python :: 3",
  "License :: OSI Approved :: MIT License",
  "Operating System :: OS Independent",
]

[project.urls]
"Homepage" = "https://github.com/yourname/dpo_financial_repo"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
