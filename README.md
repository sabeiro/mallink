---
title: "Mallink library description"
author: Giovanni Marelli
date: 2019-01-12
rights:  Creative Commons Non-Commercial Share Alike 3.0
language: en-US
output: 
	md_document:
		variant: markdown_strict+backtick_code_blocks+autolink_bare_uris+markdown_github
---

# mallink
Reinforcement learning + Monte Carlo and Markov chains for polymer/path systems

![module_mallink](docs/f_ops/module_mallink.svg "module mallink")

_overview of the mallink modules_


> computing
 
* `calc_finiteDiff.py`
	* finite difference implementation of differential equations
* `kernel_list.py`
	* collection of kernels
  

# antani

Ant - agent/network intelligence 

![antani_logo](docs/f_ops/antani_logo.svg "antani logo")

_ants optimizing paths on a network_

Antani is an agent/network based optimization engine for field operations

Content: 

## kpi

[kpi comparison](docs/antani_kpi.md)

* definition of kpis
* different kpi per run

![kpi](docs/f_ops/kpi_comparison.png "kpi comparison")

_kpi comparison_

## engine

[engine functionalities](docs/mallink_engine.md) 

* list of moves
* performances

![engine](docs/f_ops/vid_phantom.gif "engine")

_engine description_

## graph

[graph building utilities](docs/geomadi_graph.md)

* retrieving a network
* building and fixing a graph

![graph](docs/f_ops/graph_detail.png "graph detail")

_graph formation_

## concept

[basic concepts](docs/antani_concept.md)

* agent
* network optimization

![antani_concept](docs/f_ops/antani_concept.svg "antani concept")

_antani concept schema_

