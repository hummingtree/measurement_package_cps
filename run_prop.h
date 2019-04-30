// -*- mode:c++; c-basic-offset:4 -*-
#ifndef INCLUDED_RUN_PROP_H_KL3
#define INCLUDED_RUN_PROP_H_KL3

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <cassert>

//#include <util/lattice.h>
#include <alg/qpropw_arg.h>
#include <alg/array_arg.h>
#include <alg/eigcg_arg.h>

#include "prop_container.h"

namespace cps {
    class Lattice;
};

class PropSettings {
public:
    static cps::Float stop_rsd_exact;
    static cps::Float stop_rsd_inexact;
    static bool do_bc_P;
    static bool do_bc_A;
    static int num_eigcg_volume_solves;
};

void run_wall_prop(AllProp *prop_e,
                   AllProp *prop,
                   cps::IntArray &eloc, 
                   cps::Fbfm &lat,
                   cps::QPropWArg &qp_arg,
                   cps::EigCGArg *eigcg_arg,
                   int traj,
                   bool do_mres);

void run_wall_box_prop(AllProp *prop_e,
                   AllProp *prop,
                   AllProp *prop_box,
                   cps::IntArray &eloc, 
                   cps::Fbfm &lat,
                   cps::QPropWArg &qp_arg,
                   cps::QPropW4DBoxArg &box_arg,
                   cps::EigCGArg *eigcg_arg,
                   int traj,
                   bool do_mres);

void run_mom_prop(AllProp *prop_e,
                  AllProp *prop,
                  cps::IntArray &eloc,
                  cps::Fbfm &lat,
                  cps::QPropWArg &qp_arg,
                  cps::EigCGArg *eigcg_arg,
                  int traj,
                  const int mom[3]);

void run_box_prop(AllProp *prop,
                  cps::Fbfm &lat,
                  cps::QPropWArg &qp_arg,
                  cps::QPropW4DBoxArg &box_arg,
                  int traj);

void run_box_prop_test_twist_all(AllProp *prop,
                  cps::Fbfm &lat,
                  cps::QPropWArg &qp_arg,
                  cps::QPropWBoxArg &box_arg,
                  cps::EigCGArg *eigcg_arg,
                  int traj);

#endif
