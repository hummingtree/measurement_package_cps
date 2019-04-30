// -*- mode:c++; c-basic-offset:4 -*-
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <util/lattice.h>
#include <util/lattice/fbfm.h>

#include <alg/array_arg.h>
#include <alg/alg_int.h>
#include <alg/eigcg_arg.h>
#include <alg/qpropw.h>

#include <util/gjp.h>
#include <util/time_cps.h>
#include <util/verbose.h>
#include <util/qcdio.h>
#include <util/error.h>

#include <string>
#include <vector>
#include <cassert>

#include "prop_container.h"
#include "eigcg.h"
#include "my_util.h"
#include "run_mres.h"
#include "run_prop.h"

#include <util/lattice/eigcg_controller.h>
#include <alg/eigcg_arg.h>

static const char *cname = "";

USING_NAMESPACE_CPS
using namespace std;

cps::Float PropSettings::stop_rsd_exact = 1e-8;
cps::Float PropSettings::stop_rsd_inexact = 1e-4;
bool PropSettings::do_bc_P = false;
bool PropSettings::do_bc_A = true;
int PropSettings::num_eigcg_volume_solves = 2;

static void run_mres_za(const QPropW &qp, const QPropWArg &qp_arg,
                        const string &rdir, int traj)
{
    string mres_fn = rdir + "/mres_"
        + tostring(qp_arg.cg.mass) + "."
        + tostring(traj);

    string za_fn = rdir + "/za_"
        + tostring(qp_arg.cg.mass) + "."
        + tostring(traj);

    run_mres(qp, qp_arg.t, mres_fn.c_str());
    run_za(qp, qp_arg.cg.mass, qp_arg.t, za_fn.c_str());
}

// Temporary hack, solve a 4D volume source to collect low modes,
// useful for AMA.
//
// Note: How many times we solve the volume source depends on how many
// low modes we want to solve. Lattice properties also apply.
//
// For 300 low modes, 1 propagator using mixed solver will be good
// (depends on EigCGArg).
//
// On 48^3 2 solves are needed for 600 low modes.
static void collect_lowmodes(Fbfm &lat,
    QPropWArg &qp_arg,
    CommonArg &com_prop)
{
  const char* fname = "collect_lowmodes()"; 
  Float timer0 = dclock();

  lat.SetBfmArg(qp_arg.cg.mass);

  double stop_rsd = qp_arg.cg.stop_rsd;
  double true_rsd = qp_arg.cg.true_rsd;

  qp_arg.cg.stop_rsd = 1e-10;
  qp_arg.cg.true_rsd = 1e-10;

  auto inverter_type = qp_arg.cg.Inverter;
  qp_arg.cg.Inverter = EIGCG;	

  QPropW4DBoxArg vol_arg;
  for(int mu = 0; mu < 4; ++mu) {
    vol_arg.box_start[mu] = 0;
    vol_arg.box_size[mu] = GJP.Sites(mu);
    vol_arg.mom[mu] = 0;
  }

  for(unsigned i = 0; i < PropSettings::num_eigcg_volume_solves; ++i) {
    // QPropWVolSrc(lat, &qp_arg, &com_prop);
    QPropW4DBoxSrc qp_box(lat, &qp_arg, &vol_arg, &com_prop);
  }

  qp_arg.cg.stop_rsd = stop_rsd;
  qp_arg.cg.true_rsd = true_rsd;
  qp_arg.cg.Inverter = inverter_type;	

  Float timer1 = dclock();
  VRB.Result(cname, fname, "Finished collecting low modes: took %e seconds\n", timer1 - timer0);
}

#if 0

extern MPI_Comm QMP_COMM_WORLD;

static void read_eigcg_eigenvectors(Fbfm& lat, const std::string& dir){
	char node_path[1024];
	sprintf(node_path, "%s/%02d/%010d", dir.c_str(), UniqueID()/(256/32), UniqueID());
	FILE* fp = fopen(node_path, "rb");
	assert(fp != NULL);
	
	EigCGController<float>* p_eigcg_ctrl = EigCGController<float>::getInstance();
	
	assert(p_eigcg_ctrl->nx == 50);

	if(not UniqueID()){
		char eigenvalue_path[1024];
		sprintf(eigenvalue_path, "%s/eigen-values.txt", dir.c_str());
		FILE* fp_eigenvalues = fopen(eigenvalue_path, "r");
		assert(fp_eigenvalues != NULL);
		int dummy;
		fscanf(fp_eigenvalues, "%d", &dummy);
		for(int i = 0; i < p_eigcg_ctrl->nx; i++){
			fscanf(fp_eigenvalues, "%lf", &(p_eigcg_ctrl->Xeigenvalues[i]));
		}
	}
	MPI_Bcast(p_eigcg_ctrl->Xeigenvalues.data(), p_eigcg_ctrl->Xeigenvalues.size(), MPI_DOUBLE, 0, QMP_COMM_WORLD);
	if(UniqueID() == 100){
		for(int i = 0; i < 50; i++){
			printf("eigenvalues[%03d] = %10.8E\n", i, p_eigcg_ctrl->Xeigenvalues[i]);
		}
	}

	for(int c = 0; c < 16; c++){
		if(UniqueID()%16 == c){
			printf("node #%04d reading eigcg eigenvectors ...\n", UniqueID());
			for(int i = 0; i < 50; i++){
				fread(p_eigcg_ctrl->getX(i), p_eigcg_ctrl->get_vec_len()*sizeof(float), 1, fp);	
			}
		}
	}

//	GJP.Tbc(BND_CND_APRD); 
//	GJP.Xbc(BND_CND_APRD); 
//	lat.BondCond(); // BondCond() is virtual so gauge field will imported to bfm


	lat.bf.GeneralisedFiveDimEnd();
	lat.bf.GeneralisedFiveDimInit();
	lat.bf.comm_init();

	for(int i = 0; i < 50; i++){
	// Test the correctness of the eigenvectors and eigenvalues
#pragma omp parallel
{
		Fermion_t aa = lat.bf.threadedAllocFermion();
		Fermion_t bb = lat.bf.threadedAllocFermion();
		Fermion_t cc = lat.bf.threadedAllocFermion();

		int me = lat.bf.thread_barrier();
		lat.bf.Mprec(p_eigcg_ctrl->getX(i), aa, cc, DaggerNo); // aa = M * eigenvector
		lat.bf.Mprec(aa, bb, cc, DaggerYes);    // bb = Mdag * aa
		lat.bf.axpby(cc, bb, p_eigcg_ctrl->getX(i), +1., -1.*p_eigcg_ctrl->Xeigenvalues[i]);
		std::complex<double> matrix_element = lat.bf.inner(bb, p_eigcg_ctrl->getX(i));
		std::complex<double> error = lat.bf.inner(cc, cc);
		std::complex<double> norm = lat.bf.inner(p_eigcg_ctrl->getX(i), p_eigcg_ctrl->getX(i));
	
		if(lat.bf.isBoss() && !me) {
			printf("#%04d: matrix_ele = %10.8E; ", i, matrix_element.real());
			printf("error_norm = %10.8E; ", error.real());
			printf("norm_eigen = %10.8E\n; ", norm.real());
//			printf("eigenvalue = %10.8E.\n", eigenvalues[i]);
		}	
		
		lat.bf.threadedFreeFermion(aa);
		lat.bf.threadedFreeFermion(bb);
		lat.bf.threadedFreeFermion(cc);
}
	}
 	
//	lat.BondCond();  	
//	GJP.Tbc(BND_CND_PRD); 
//	GJP.Xbc(BND_CND_PRD); 


	return;
}


static void diagonalize_eigcg_eigenvectors(bfm_evo<float>& bf){
	// Assuming single thread.
	using namespace Eigen;

	typedef Eigen::Matrix<std::complex<double>, Dynamic, Dynamic, Eigen::RowMajor> SubMat;

	EigCGController<float>* p_eigcg_ctrl = EigCGController<float>::getInstance();

    const int N = p_eigcg_ctrl->get_max_def_len();

	SubMat H(N,N);
	SubMat G(N,N);

    memcpy(H.data(), p_eigcg_ctrl->h, N*N*sizeof(std::complex<double>));
    memcpy(G.data(), p_eigcg_ctrl->G, N*N*sizeof(std::complex<double>));

// diagonalize both matrices.

	SelfAdjointEigenSolver<SubMat> eigG(G);

	SubMat V1(N,N);
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
			assert(eigG.eigenvalues()[i] > 0.);
			if(i == j) V1(i,j) = 1./sqrt(eigG.eigenvalues()[i]);
			else V1(i,j) = 0.;
		}
        if(not UniqueID()) printf("D0[%d] = %.8E\n", i, eigG.eigenvalues()[i]);
    }
    
    SubMat V2(eigG.eigenvectors()*V1);
    H = V2.adjoint() * H * V2;

	SelfAdjointEigenSolver<SubMat> eigH(H);

    for(int i = 0; i < N; i++){
		if(not UniqueID()) printf("H1[%d] = %.8E\n", i, eigH.eigenvalues()[i]);
    }
	if(not UniqueID()) printf("p_eigcg_ctrl->nx = %d\n", p_eigcg_ctrl->nx);
	for(int i = 0; i < p_eigcg_ctrl->nx; i++){
		p_eigcg_ctrl->Xeigenvalues[i] = eigH.eigenvalues()[i];
	}

    for(int i = 0; i < p_eigcg_ctrl->nx; i++){
#pragma omp parallel
{
		bf.set_zero(p_eigcg_ctrl->getX(i));
}
		for(int l = 0; l < N; l++){
            qlat::Complex c = 0.;
            for(int j = 0; j < N; j++){
                c += V2(l,j) * eigH.eigenvectors()(j,i);
            }
#pragma omp parallel
{
            bf.caxpy(p_eigcg_ctrl->getX(i), p_eigcg_ctrl->getU(l), p_eigcg_ctrl->getX(i), c.real(), c.imag());
}
		}
    }
}
#endif

EigCG* eigcg_init(Fbfm &lat,
    QPropWArg &qp_arg,
    EigCGArg *eigcg_arg,
    CommonArg &com_prop,
    const char* bc_str,
    int traj)
{
  const char* fname = "eigcg_init()";
  VRB.Result(cname, fname, "Doing EigCG setup (collecting low modes) for mf = %.6f\n", qp_arg.cg.mass);

  assert(eigcg_arg != NULL);

  int Ls = Fbfm::arg_map[qp_arg.cg.mass].Ls;
  if (Fbfm::madwf_arg_map.count(qp_arg.cg.mass) != 0) {
    // If we are using MADWF, EigCG should collect low modes for the cheap inner solve.
    Ls = Fbfm::madwf_arg_map[qp_arg.cg.mass].cheap_approx.Ls;
  }

  EigCG *eig_cg = new EigCG(eigcg_arg, Ls, Fbfm::use_mixed_solver, 0);

  EigCGController<float>* p_eigcg_ctrl = EigCGController<float>::getInstance();	
  //	p_eigcg_ctrl->init_cg_only = true;	

  collect_lowmodes(lat, qp_arg, com_prop);

  // make sure we collected enough low modes
  if (eig_cg->get_num_low_modes_collected() != eigcg_arg->max_def_len) {
    ERR.General(cname, fname, "Didn't collect enough low modes. Only got %d out of a desired %d. "
        "Try increasing PropSettings::num_eigcg_volume_solves from %d to something higher.\n", 
        eig_cg->get_num_low_modes_collected(), eigcg_arg->max_def_len, PropSettings::num_eigcg_volume_solves);
  } else {
    VRB.Result(cname, fname, "Finished EigCG setup: collected %d low modes.\n", eigcg_arg->max_def_len);
  }

  // The following diagonalize the eigcg eigenvectors
#if 0
  diagonalize_eigcg_eigenvectors(lat.bf);

  std::string luchang_dir = "/bgusr/home/jtu/eigenvectors/job-01200-lanczos.output";
  read_eigcg_eigenvectors(lat, luchang_dir);

  lat.SetBfmArg(qp_arg.cg.mass);

  lat.bf.GeneralisedFiveDimEnd();
  lat.bf.GeneralisedFiveDimInit();
  lat.bf.comm_init();

  // Check the quality of the eigenvectors
  for(int i = 0; i < p_eigcg_ctrl->nx; i++){
#pragma omp parallel
    {
      Fermion_t a = lat.bf.threadedAllocFermion();
      Fermion_t b = lat.bf.threadedAllocFermion();
      Fermion_t c = lat.bf.threadedAllocFermion();

      int me = lat.bf.thread_barrier();

      Fermion_t x = p_eigcg_ctrl->getX(i);

      // normalize x.
      std::complex<double> xx = lat.bf.inner(x, x);
      lat.bf.axpby(x, x, x, 1./sqrt(xx.real()), 0.);
      if(lat.bf.isBoss() && !me) {
        printf("xx = %10.8E.\n", xx.real());
      }	

      xx = lat.bf.inner(x, x);

      lat.bf.Mprec(x, a, c, DaggerNo); // a = M * v
      lat.bf.Mprec(a, b, c, DaggerYes); // b = Mdag * a
      std::complex<double> xMdagMx = lat.bf.inner(x, b);

      lat.bf.axpby(c, b, x, 1., -1.*xMdagMx.real());

      std::complex<double> cc = lat.bf.inner(c, c);

      if(lat.bf.isBoss() && !me) {
        printf("#%04d: xMdagMx = %10.8E; ", i, xMdagMx.real());
        printf("|MdagMx - xMdagMx*x|^2 = %10.8E; ", cc.real());
        printf("xx = %10.8E.\n", xx.real());
      }

      lat.bf.threadedFreeFermion(a);
      lat.bf.threadedFreeFermion(b);
      lat.bf.threadedFreeFermion(c);
    }
  }		
#endif

  char fn[512];
  sprintf(fn, "../results%s/eigH_%0.5f.%d", bc_str, qp_arg.cg.mass, traj);
  eig_cg->printH(fn);
  sprintf(fn, "../results%s/eigG_%0.5f.%d", bc_str, qp_arg.cg.mass, traj);
  eig_cg->printG(fn);

  return eig_cg;
}

void eigcg_cleanup(EigCG *eig_cg)
{
    delete eig_cg;
}

void run_wall_prop(AllProp *prop_e,
                   AllProp *prop,
                   IntArray &eloc, 
                   Fbfm &lat,
                   QPropWArg &qp_arg,
                   EigCGArg *eigcg_arg,
                   int traj,
                   bool do_mres)
{
    const char *fname = "run_wall_prop()";

    // Check boundary condition. We need this to ensure that we are
    // doing P + A and P - A, not A + P and A - P (I think it's OK to
    // skip this check, though).
    if(GJP.Tbc() == BND_CND_APRD) {
        ERR.General(cname, fname, "Boundary condition does not match!\n");
    }

    lat.SetBfmArg(qp_arg.cg.mass);
    
	char buf[256];
    CommonArg com_prop;
    sprintf(buf, "../results/%s.%d", qp_arg.ensemble_label, traj);
    com_prop.set_filename(buf);

    // David: only need A 
    for(int bc = 1; bc < 2; ++bc) {
        GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
        lat.BondCond();

        EigCG *eig_cg = NULL;
	    if (qp_arg.cg.Inverter == EIGCG) {
	        eig_cg = eigcg_init(lat, qp_arg, eigcg_arg, com_prop, (bc == 0 ? "EP" : "EA"), traj);
	    }
//        if(eigcg_arg) {
//            eig_cg = new EigCG(eigcg_arg, Fbfm::use_mixed_solver);
//            VRB.Result(cname, fname, "Collecting low modes...\n");
//            collect_lowmodes(lat, qp_arg, com_prop);
//            
//            const string fn = string("../results") + (bc == 0 ? "EP" : "EA")
//                + "/eigH_" + (do_mres ? "wall_" : "twist_")
//                + tostring(qp_arg.cg.mass) + "."
//                + tostring(traj);
//
//            eig_cg->printH(fn);
//        }

        Float timer0 = dclock();

        // exact propagators
        if(prop_e != NULL) {
            double stop_rsd = qp_arg.cg.stop_rsd;
            double true_rsd = qp_arg.cg.true_rsd;
            
            qp_arg.cg.stop_rsd = 1e-8;
            qp_arg.cg.true_rsd = 1e-8;
            for(unsigned i = 0; i < eloc.v.v_len; ++i) {
                qp_arg.t = eloc.v.v_val[i];
                VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", qp_arg.t, qp_arg.cg.stop_rsd);

                QPropWWallSrc qp_wall(lat, &qp_arg, &com_prop);
                if(do_mres) {
                    run_mres_za(qp_wall, qp_arg,
                                string("../results") + (bc == 0 ? "EP" : "EA"),
                                traj);
                }
                prop_e->add(qp_wall, qp_arg.t, bc == 0);
            }
            qp_arg.cg.stop_rsd = stop_rsd;
            qp_arg.cg.true_rsd = true_rsd;
        }

        Float timer1 = dclock();

        // inexact propagators
        for(int t = 0; t < GJP.Sites(3); t+=4) {
            qp_arg.t = t;
            VRB.Result(cname, fname, "Solving inexact propagator at %d with stop_rsd = %e\n", qp_arg.t, qp_arg.cg.stop_rsd);
            QPropWWallSrc qp_wall(lat, &qp_arg, &com_prop);
            if(do_mres) {
                run_mres_za(qp_wall, qp_arg,
                            string("../results") + (bc == 0 ? "P" : "A"),
                            traj);
            }
            prop->add(qp_wall, qp_arg.t, bc == 0);
        }

        Float timer2 = dclock();

        VRB.Result(cname, fname, "Total time for   exact propagators = %e seconds\n", timer1 - timer0);
        VRB.Result(cname, fname, "Total time for inexact propagators = %e seconds\n", timer2 - timer1);

        delete eig_cg;
        lat.BondCond();
    }

    // Note: If I call lat.BondCond() even times, then there is no
    // overall effect.
    GJP.Tbc(BND_CND_PRD);
}

void run_wall_box_prop(AllProp *prop_e,
    AllProp *prop,
    AllProp *prop_box,
    cps::IntArray &eloc, 
    cps::Fbfm &lat,
    cps::QPropWArg &qp_arg,
    cps::QPropW4DBoxArg &box_arg,
    cps::EigCGArg *eigcg_arg,
    int traj,
    bool do_mres)
{
  const char *fname = "run_wall_box_prop()";

  // Check boundary condition. We need this to ensure that we are
  // doing P + A and P - A, not A + P and A - P (I think it's OK to
  // skip this check, though).
  if(GJP.Tbc() == BND_CND_APRD) {
    ERR.General(cname, fname, "Boundary condition does not match!\n");
  }

  lat.SetBfmArg(qp_arg.cg.mass);

  char buf[256];
  CommonArg com_prop;
  sprintf(buf, "../results/%s.%d", qp_arg.ensemble_label, traj);
  com_prop.set_filename(buf);

  // David: only need A 
  for(int bc = 1; bc < 2; ++bc) {
    GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
    lat.BondCond();

    EigCG *eig_cg = NULL;
    if (qp_arg.cg.Inverter == EIGCG) {
      eig_cg = eigcg_init(lat, qp_arg, eigcg_arg, com_prop, (bc == 0 ? "EP" : "EA"), traj);
    }

    Float timer0 = dclock();

    // exact propagators
    if(prop_e != NULL) {
      double stop_rsd = qp_arg.cg.stop_rsd;
      double true_rsd = qp_arg.cg.true_rsd;

      qp_arg.cg.stop_rsd = 1e-8;
      qp_arg.cg.true_rsd = 1e-8;
      for(unsigned i = 0; i < eloc.v.v_len; ++i) {
        qp_arg.t = eloc.v.v_val[i];
        VRB.Result(cname, fname, "Solving exact propagator at %d with stop_rsd = %e\n", qp_arg.t, qp_arg.cg.stop_rsd);

        QPropWWallSrc qp_wall(lat, &qp_arg, &com_prop);
        if(do_mres) {
          run_mres_za(qp_wall, qp_arg,
              string("../results") + (bc == 0 ? "EP" : "EA"),
              traj);
        }
        prop_e->add(qp_wall, qp_arg.t, bc == 0);
      }
      qp_arg.cg.stop_rsd = stop_rsd;
      qp_arg.cg.true_rsd = true_rsd;
    }

    Float timer1 = dclock();

    // inexact propagators
    for(int t = 0; t < GJP.Sites(3); ++t) {
      qp_arg.t = t;
      VRB.Result(cname, fname, "Solving inexact propagator at %d with stop_rsd = %e\n", qp_arg.t, qp_arg.cg.stop_rsd);
      QPropWWallSrc qp_wall(lat, &qp_arg, &com_prop);
      if(do_mres) {
        run_mres_za(qp_wall, qp_arg,
            string("../results") + (bc == 0 ? "P" : "A"),
            traj);
      }
      prop->add(qp_wall, qp_arg.t, bc == 0);
    }

    Float timer2 = dclock();

    VRB.Result(cname, fname, "Total time for   exact propagators = %4.2e seconds\n", timer1 - timer0);
    VRB.Result(cname, fname, "Total time for inexact propagators = %4.2e seconds\n", timer2 - timer1);
    
    // The box propagator.
    for(int t = 0; t < GJP.Sites(3); ++t) {
        box_arg.box_start[3] = qp_arg.t = t;
        VRB.Result(cname, fname, "Solving     box propagator at %d with stop_rsd = %e\n", qp_arg.t, qp_arg.cg.stop_rsd);
        QPropWZ3BWallSrc qp_z3(lat, &qp_arg, &box_arg, &com_prop);
        prop_box->add(qp_z3, box_arg.box_start[3], bc == 0);
    }
  
    Float timer3 = dclock();
    VRB.Result(cname, fname, "Total time for     box propagators = %4.2e seconds\n", timer3 - timer2);

    delete eig_cg;
    lat.BondCond();
  }

  // Note: If I call lat.BondCond() even times, then there is no
  // overall effect.
  GJP.Tbc(BND_CND_PRD);
}

void run_mom_prop(AllProp *prop_e,
                  AllProp *prop,
                  IntArray &eloc,
                  Fbfm &lat,
                  QPropWArg &qp_arg,
                  EigCGArg *eigcg_arg,
                  int traj,
                  const int mom[3])
{
    const char *fname = "run_mom_prop()";

    // Ensure that all 4 directions have periodic boundary condition.
    // FIXME: This check is not perfect as we have no way detecting
    // how the actual gauge field data were manipulated.
    for(int mu = 0; mu < 4; ++mu) {
        if(GJP.Bc(mu) == BND_CND_APRD) {
            ERR.General(cname, fname, "Boundary condition does not match!\n");
        }
        if(mu < 3 && mom[mu]) {
            GJP.Bc(mu, BND_CND_APRD);
        }
    }

    lat.SetBfmArg(qp_arg.cg.mass);
    
	char buf[256];
    CommonArg com_prop;
    sprintf(buf, "../results/%s.%d", qp_arg.ensemble_label, traj);
    com_prop.set_filename(buf);

    // P+A
    for(int bc = 1; bc < 2; ++bc) {
        GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
        lat.BondCond();

        EigCG *eig_cg = NULL;
	    if (qp_arg.cg.Inverter == EIGCG) {
	        eig_cg = eigcg_init(lat, qp_arg, eigcg_arg, com_prop, (bc == 0 ? "EP" : "EA"), traj);
	    }

//        EigCG *eig_cg = NULL;
//        if(eigcg_arg) {
//            eig_cg = new EigCG(eigcg_arg, Fbfm::use_mixed_solver);
//            VRB.Result(cname, fname, "Collecting low modes...\n");
//            collect_lowmodes(lat, qp_arg, com_prop);
//
//            const string fn = string("../results") + (bc == 0 ? "EP" : "EA")
//                + "/eigH_mom_" + tostring(qp_arg.cg.mass) + "."
//                + tostring(traj);
//
//            eig_cg->printH(fn);
//        }

        // exact propagators
        if(prop_e != NULL) {
            double stop_rsd = qp_arg.cg.stop_rsd;
            double true_rsd = qp_arg.cg.true_rsd;

            qp_arg.cg.stop_rsd = 1e-8;
            qp_arg.cg.true_rsd = 1e-8;
            for(unsigned i = 0; i < eloc.v.v_len; ++i) {
                qp_arg.t = eloc.v.v_val[i];
                VRB.Result(cname, fname, "Solving exact propagator at %d\n", qp_arg.t);

                QPropWMomCosTwistSrc qp_mom(lat, &qp_arg, mom, &com_prop);
                prop_e->add(qp_mom, qp_arg.t, bc == 0);
            }
            qp_arg.cg.stop_rsd = stop_rsd;
            qp_arg.cg.true_rsd = true_rsd;
        }

        // inexact propagators
        for(int t = 0; t < GJP.Sites(3); ++t) {
            qp_arg.t = t;
            QPropWMomCosTwistSrc qp_mom(lat, &qp_arg, mom, &com_prop);
            prop->add(qp_mom, qp_arg.t, bc == 0);
        }

        delete eig_cg;
        lat.BondCond();
    }

    // Note: If I call lat.BondCond() even times, then there is no
    // overall effect.
    for(int mu = 0; mu < 4; ++mu) {
        GJP.Bc(mu, BND_CND_PRD);
    }
}

void run_box_prop(AllProp *prop,
                  Fbfm &lat,
                  QPropWArg &qp_arg,
                  QPropW4DBoxArg &box_arg,
                  int traj)
{
    const char *fname = "run_box_prop()";

    // Check boundary condition. We need this to ensure that we are
    // doing P + A and P - A, not A + P and A - P (I think it's OK to
    // skip this check, though).
    if(GJP.Tbc() == BND_CND_APRD) {
        ERR.General(cname, fname, "Boundary condition does not match!\n");
    }

    lat.SetBfmArg(qp_arg.cg.mass);

    char buf[256];
    CommonArg com_prop;
    sprintf(buf, "../results/%s.%d", qp_arg.ensemble_label, traj);
    com_prop.set_filename(buf);

    // A only
    for(int bc = 1; bc < 2; ++bc) {
        GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
        lat.BondCond();

        // inexact propagators
        for(int t = 0; t < GJP.Sites(3); ++t) {
            box_arg.box_start[3] = qp_arg.t = t;
            QPropWZ3BWallSrc qp_z3(lat, &qp_arg, &box_arg, &com_prop);

//            run_mres_za(qp_z3, qp_arg,
//                        string("../results") + (bc == 0 ? "P" : "A"),
//                        traj);

            prop->add(qp_z3, box_arg.box_start[3], bc == 0);
        }

        lat.BondCond();
    }

    // Note: If I call lat.BondCond() even times, then there is no
    // overall effect.
    GJP.Tbc(BND_CND_PRD);
}

void run_box_prop_test_twist_all(AllProp *prop,
                  Fbfm &lat,
                  QPropWArg &qp_arg,
                  QPropWBoxArg &box_arg,
				  EigCGArg *eigcg_arg,
                  int traj)
{
    const char *fname = "run_box_prop()";
    
	double stop_rsd = qp_arg.cg.stop_rsd;
    double true_rsd = qp_arg.cg.true_rsd;

    qp_arg.cg.stop_rsd = 1e-8;
    qp_arg.cg.true_rsd = 1e-8;
    
	// Check boundary condition. We need this to ensure that we are
    // doing P + A and P - A, not A + P and A - P (I think it's OK to
    // skip this check, though).
    if(GJP.Tbc() == BND_CND_APRD) {
        ERR.General(cname, fname, "Boundary condition does not match!\n");
    }
    
// TODO: For test purposes twist all three spatial directions
//	for(int mu = 0; mu < 3; ++mu) {
//        GJP.Bc(mu, BND_CND_APRD);
//        GJP.Bc(mu, BND_CND_APRD);
//    }
// End test

// Twist the x direction
	GJP.Bc(0, BND_CND_APRD);

	lat.SetBfmArg(qp_arg.cg.mass);

    char buf[256];
    CommonArg com_prop;
    sprintf(buf, "../results/%s.%d", qp_arg.ensemble_label, traj);
    com_prop.set_filename(buf);

    // A only
    for(int bc = 1; bc < 2; ++bc) {
        GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
        lat.BondCond();
	
//		lat.SetBfmArg(qp_arg.cg.mass);
//
//		lat.bf.GeneralisedFiveDimEnd();
//		lat.bf.GeneralisedFiveDimInit();
//		lat.bf.comm_init();

		if(GJP.Xbc() == BND_CND_APRD) qlat::Printf("Boundary contion in x direction: Anti-periodic.\n");
		else						 qlat::Printf("Boundary contion in x direction:      periodic.\n");

		if(GJP.Ybc() == BND_CND_APRD) qlat::Printf("Boundary contion in y direction: Anti-periodic.\n");
		else						 qlat::Printf("Boundary contion in y direction:      periodic.\n");

		if(GJP.Zbc() == BND_CND_APRD) qlat::Printf("Boundary contion in z direction: Anti-periodic.\n");
		else						 qlat::Printf("Boundary contion in z direction:      periodic.\n");

		if(GJP.Tbc() == BND_CND_APRD) qlat::Printf("Boundary contion in t direction: Anti-periodic.\n");
		else						 qlat::Printf("Boundary contion in t direction:      periodic.\n");
		
		EigCG *eig_cg = NULL;
	    eig_cg = eigcg_init(lat, qp_arg, eigcg_arg, com_prop, (bc == 0 ? "EP" : "EA"), traj);
// TODO: destructor of lat will call bf.end() ... ish. Need to restart communication here.

        // inexact propagators
//        for(int t = 0; t < GJP.Sites(3); ++t) {
        for(int t = 0; t < 1; ++t) {
            qp_arg.t = t;
			for(int it = 4800; it < 4801; it += 200){
         		qp_arg.cg.max_num_iter = it;
                VRB.Result(cname, fname, "cg.max_num_iter = %d\n", it);
				QPropWWallSrc qp_wall(lat, &qp_arg, &com_prop);
//        		QPropWBoxSrc qp_z3(lat, &qp_arg, &box_arg, &com_prop);
			}
//            run_mres_za(qp_z3, qp_arg,
//                        string("../results") + (bc == 0 ? "P" : "A"),
//                        traj);

//            prop->add(qp_z3, t, bc == 0);
        }

        lat.BondCond();
    }

/*

// TODO: For test purposes twist all three spatial directions
// Now change back.
	for(int mu = 0; mu < 3; ++mu) {
        GJP.Bc(mu, BND_CND_PRD);
    }
// End test

	// TODO: Do this for untwisted situation
	// A only
    for(int bc = 1; bc < 2; ++bc) {
        GJP.Tbc(bc == 0 ? BND_CND_PRD : BND_CND_APRD);
        lat.BondCond();

        // inexact propagators
        for(int t = 0; t < GJP.Sites(3); ++t) {
            qp_arg.t = t;
            QPropWBoxSrc qp_z3(lat, &qp_arg, &box_arg, &com_prop);

//            run_mres_za(qp_z3, qp_arg,
//                        string("../results") + (bc == 0 ? "P" : "A"),
//                        traj);

//            prop->add(qp_z3, t, bc == 0);
        }

        lat.BondCond();
    }

	qp_arg.cg.stop_rsd = stop_rsd;
	qp_arg.cg.true_rsd = true_rsd;

    // Note: If I call lat.BondCond() even times, then there is no
    // overall effect.
    GJP.Tbc(BND_CND_PRD);

*/

}
