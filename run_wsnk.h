// -*- mode:c++; c-basic-offset:4 -*-
#ifndef INCLUDED_RUN_WSNK_H_KL3
#define INCLUDED_RUN_WSNK_H_KL3

#include <vector>
#include <alg/qpropw.h>
#include "prop_container.h"

CPS_START_NAMESPACE

// compute wall sink propagator, with momentum
void run_wall_snk(std::vector<std::vector<WilsonMatrix> > *wsnk,
                  const AllProp &prop,
                  PROP_TYPE ptype, const int *p = NULL);

CPS_END_NAMESPACE

#endif
