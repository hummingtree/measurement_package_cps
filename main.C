#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include <util/lattice.h>
#include <util/lattice/fbfm.h>

#include <alg/array_arg.h>
#include <alg/common_arg.h>
#include <alg/eigcg_arg.h>
#include <alg/alg_fix_gauge.h>
#include <alg/qpropw_arg.h>
#include <alg/meas_arg.h>

#include <alg/alg_plaq.h>

#include <util/gjp.h>
#include <util/time_cps.h>
#include <util/verbose.h>
#include <util/error.h>
#include <util/qcdio.h>
#include <util/ReadLatticePar.h>

#include <string>
#include <vector>
#include <cassert>

#include <qmp.h>

#include "prop_container.h"
#include "eigcg.h"
#include "my_util.h"
#include "run_kl3.h"
#include "run_2pion.h"
#include "run_k2pipi.h"
#include "run_bk.h"
#include "run_mres.h"
#include "run_meson.h"
#include "run_omega.h"
#include "run_prop.h"
#include "twisted_bc.h"

#include <util/eigen_container.h>

#include <qlat/qlat.h>

// BG/Q Machine uses Big endianness
#define __BYTE_ORDER __BIG_ENDIAN

static const char *cname = "";

USING_NAMESPACE_CPS
using namespace std;

DoArg do_arg;
DoArgExt doext_arg;

MeasArg meas_arg;
QPropWArg lqpropw_arg;
QPropWArg sqpropw_arg;
QPropW4DBoxArg box_arg;
QPropWBoxArg box_arg_3d;
FixGaugeArg fix_gauge_arg;
EigCGArg l_eigcg_arg;
EigCGArg s_eigcg_arg;

// Lanczos
LanczosArg lanczos_arg;

// Integer arrays, for setting time locations of exact propagators
IntArray l_ut_loc; // untwisted light
IntArray l_tw_loc; // twisted light 
IntArray s_ut_loc; // untwisted strange
IntArray s_tw_loc; // twisted strange

// d quark momentum for K -> pi pi, for 48^3 this should be {1, 1, 1}.
IntArray d_mom_kpp;

// l and s quark twists
FloatArray l_twist_arg;
FloatArray s_twist_arg;

NoArg no_arg;
void measure_plaq(Lattice& lat)
{
	const char* fname = "measure_plaq()";
	char dummy[] = "dummy_file.dat";
	CommonArg common_arg;
	common_arg.set_filename(dummy);
	AlgPlaq plaq(lat, &common_arg, &no_arg);
	plaq.run();
	return;
}

#define decode_vml(arg_name)  do{                                       \
	if ( ! arg_name.Decode(#arg_name".vml", #arg_name) )            \
	ERR.General(cname, fname, "Bad " #arg_name ".vml.\n");      \
} while(0)  

void decode_vml_all(void)
{
	const char *fname = "decode_vml_all()";

	decode_vml(do_arg);
	decode_vml(doext_arg);
	
	decode_vml(meas_arg);
	decode_vml(lqpropw_arg);
	decode_vml(sqpropw_arg);
	decode_vml(box_arg);
	decode_vml(box_arg_3d);
	decode_vml(fix_gauge_arg);
	decode_vml(l_eigcg_arg);
	decode_vml(s_eigcg_arg);

	decode_vml(l_ut_loc);
	decode_vml(l_tw_loc);
	decode_vml(s_ut_loc);
	decode_vml(s_tw_loc);

	decode_vml(d_mom_kpp);

	decode_vml(l_twist_arg);
	decode_vml(s_twist_arg);

	decode_vml(lanczos_arg);
}

#undef encode_vml
#define encode_vml(arg_name, traj) do{                          \
	char vml_file[256];                                           \
	sprintf(vml_file, #arg_name".%d", traj);                      \
	if(!arg_name.Encode(vml_file, #arg_name)){                    \
		ERR.General(cname, fname, #arg_name " encoding failed.\n"); \
	}                                                             \
}while(0)

void encode_vml_all(void)
{
	const char *fname = "encode_vml_all()";

	encode_vml(do_arg, 0);
	encode_vml(doext_arg, 0);
	
	encode_vml(meas_arg, 0);
	encode_vml(lqpropw_arg, 0);
	encode_vml(sqpropw_arg, 0);
	encode_vml(box_arg, 0);
	encode_vml(box_arg_3d, 0);
	encode_vml(fix_gauge_arg, 0);
	encode_vml(l_eigcg_arg, 0);
	encode_vml(s_eigcg_arg, 0);

	encode_vml(l_ut_loc, 0);
	encode_vml(l_tw_loc, 0);
	encode_vml(s_ut_loc, 0);
	encode_vml(s_tw_loc, 0);

	encode_vml(d_mom_kpp, 0);

	encode_vml(l_twist_arg, 0);
	encode_vml(s_twist_arg, 0);
	
	encode_vml(lanczos_arg, 0);
}

void load_checkpoint(Lattice& lat, int traj);
void init_bfm(int *argc, char **argv[]);
void setup(int argc, char *argv[]);

void run_ritz(GnoneFbfm& lattice){

	lattice.ImportGauge();
	GJP.Tbc(BND_CND_APRD); 
	GJP.Xbc(BND_CND_APRD); 
	lattice.BondCond();

	lattice.bd.GeneralisedFiveDimEnd();
	lattice.bd.GeneralisedFiveDimInit();
	lattice.bd.comm_init();

    lattice.SetBfmArg(lqpropw_arg.cg.mass);

	lattice.bd.residual = 1E-8;
	lattice.bd.max_iter = 200000;

	size_t f_dim = GJP.VolNodeSites () * lattice.FsiteSize () / 2;

#pragma omp parallel
	{
		Fermion_t x = lattice.bd.threadedAllocFermion();

#pragma omp for
		for(size_t index = 0; index < f_dim; index++){
			((double*)x)[index] = 1.;
		}

		double eigenvalue = lattice.bd.ritz(x, 1); // 1 for computing minimum
		Fermion_t aa = lattice.bd.threadedAllocFermion();
		Fermion_t bb = lattice.bd.threadedAllocFermion();
		Fermion_t cc = lattice.bd.threadedAllocFermion();

		int me = lattice.bd.thread_barrier();
		lattice.bd.Mprec(x, aa, cc, DaggerNo); // aa = M * eigenvector
		lattice.bd.Mprec(aa, bb, cc, DaggerYes);    // bb = Mdag * aa
		std::complex<double> matrix_element = lattice.bd.inner(bb, x);
		std::complex<double> norm = lattice.bd.inner(x, x);
		lattice.bd.axpby(cc, bb, x, 1./matrix_element.real(), -1.);
		std::complex<double> c2 = lattice.bd.inner(cc, cc);
		if(lattice.bd.isBoss() && !me) {
			// char output_file[1024];
			// sprintf(output_file, "Ritz.%03d", UniqueID());
			// FILE* p = fopen(output_file, "w");
		    // for(size_t index = 0; index < f_dim/2; index++){
		    //    fprintf(p, "%.6E %.6E\n", ((float*)(x))[index*2], ((float*)(x))[index*2+1]);
		    // }

			printf("matrix_ele = %.12e + I %.12e\n", matrix_element.real(), matrix_element.imag());
			printf("norm_eigen = %.12e + I %.12e\n", norm.real(), norm.imag());
			printf("c2 = %.12e + I %.12e\n", c2.real(), c2.imag());
			printf("eigenvalue = %.12e\n", eigenvalue);
		}	

		lattice.bd.threadedFreeFermion(aa);
		lattice.bd.threadedFreeFermion(bb);
		lattice.bd.threadedFreeFermion(cc);
	}
 
	lattice.BondCond();
   	GJP.Tbc(BND_CND_PRD); 

}

long load_eigenvectors_luchang(GnoneFbfm& lattice, const std::string& old_path,
	multi1d<Fermion_t[2]>& eigenvectors, multi1d<float>& eigenvalues)
{
	using namespace qlat;
	
	TIMER_VERBOSE_FLOPS("load_eigenvectors_luchang()");
    long total_bytes = 0;

	std:vector<double> tmp_eigenvalues = read_eigen_values(old_path);
	qassert(eigenvalues.size() <= tmp_eigenvalues.size()); // should NOT ask for more than we actually have
	for(int i = 0; i < eigenvalues.size(); i++){
		eigenvalues[i] = tmp_eigenvalues[i];
		if(not UniqueID()){
			printf("eigenvalue #%04d = %.12E\n", i, eigenvalues[i]);
		}
	}

	const CompressedEigenSystemInfo cesi = read_compressed_eigen_system_info(old_path);
	
	qlat::Coordinate size_node(GJP.Xnodes(), GJP.Ynodes(), GJP.Znodes(), GJP.Tnodes());
	
//	const int idx = index_from_coordinate(qlat::Coordinate(CoorX(), CoorY(), CoorZ(), CoorT()), size_node);
	const int idx = qlat::get_id_node();

	const int idx_size = product(size_node);
	const int dir_idx = compute_dist_file_dir_id(idx, idx_size);

	CompressedEigenSystemBases cesb;
	CompressedEigenSystemCoefs cesc;
	
	init_compressed_eigen_system_bases(cesb, cesi, idx, size_node);
	init_compressed_eigen_system_coefs(cesc, cesi, idx, size_node);

//	int node_rank;
//	MPI_Comm_rank(QMP_COMM_WORLD, &node_rank);
//	printf("Node #%03d(CPS): %02dx%02dx%02dx%02d ; #%03d(QMP_COMM_WORLD) ; #%03d(qlat) loading eigenvectors\n", UniqueID(), CoorX(), CoorY(), CoorZ(), CoorT(), node_rank, qlat::get_id_node());

/*
	std::vector<crc32_t> crcs;
	const int n_cycle = std::max(1, get_num_node() / 32); // TODO: should change 8 to some systematic number
	{
		long bytes = 0;
		for (int i = 0; i < n_cycle; ++i) {
			TIMER_VERBOSE_FLOPS("load_compressed_eigen_vectors-load-cycle");
			if (get_id_node() % n_cycle == i) {
				crcs = load_node(cesb, cesc, cesi, old_path);
				bytes = get_data(cesb).data_size() + get_data(cesc).data_size();
			} else {
				bytes = 0;
			}
			glb_sum(bytes);
			displayln_info(fname + ssprintf(": cycle / n_cycle = %4d / %4d", i + 1, n_cycle));
			timer.flops += bytes;
			total_bytes += bytes;
		}
		timer.flops += total_bytes;
	}

*/

	std::vector<crc32_t> crcs = load_node(cesb, cesc, cesi, old_path);
	qlat::sync_node();

	std::vector<BlockedHalfVector> bhvs;
	decompress_eigen_system(bhvs, cesb, cesc);
	std::vector<HalfVector> hvs;
	convert_half_vectors(hvs, bhvs);

	const long size = hvs[0].field.size();
  	std::vector<std::complex<float> > buffer(size);

	// change to bfm format
	for(int i = 0; i < (int)hvs.size(); ++i) {
//	for(int i = 0; i < 2; ++i) {
		TIMER_FLOPS("converting to bfm format");
//		timer.flops += get_data(hvs[i]).data_size();
		qassert(hvs[i].geo.is_only_local());
		for(long m = 0; m < size/2; ++m) {
			buffer[m*2] = hvs[i].field[m];
			buffer[m*2+1] = hvs[i].field[size/2+m];
		}
		
		hvs[i].init(); // clear memory for hvs

		eigenvectors[i][1] = new char[size*sizeof(std::complex<float>)]; // TODO: dynamical allocation!!!
		memcpy(eigenvectors[i][1], (void*)(buffer.data()), size*sizeof(std::complex<float>));
		
		float *temp = (float*)eigenvectors[i][1];
		double sum = 0.;
		for (size_t ind = 0; ind < (size_t)size*2; ind++){
			sum += temp[ind] * temp[ind];
		}
		cps::glb_sum(&sum);
		if((sum-1.)*(sum-1.) > 1E-6){
			printf("eigenvector #%04d is not normalized: norm = %.12E\n", i, sum);
		}
		if(not UniqueID()) printf("eigenvector number %d norm = %.12E\n", i, sum);
	
	}
	return 0;
}

long test_eigenvectors(GnoneFbfm& lattice,
	multi1d<Fermion_t[2]>& eigenvectors, multi1d<float>& eigenvalues, multi1d<float>& matrix_elements)
{
//	lattice.ImportGauge();
	GJP.Tbc(BND_CND_APRD); 
	// GJP.Xbc(BND_CND_APRD); 
	lattice.BondCond(); // BondCond() is virtual so gauge field will imported to bfm

	lattice.bf.GeneralisedFiveDimEnd();
	lattice.bf.GeneralisedFiveDimInit();
	lattice.bf.comm_init();

//	char output_file[1024];
//	sprintf(output_file, "Chritoph.%03d", UniqueID());
//	FILE* p = fopen(output_file, "w");
//    size_t f_dim = GJP.VolNodeSites () * lattice.FsiteSize () / 2;
//    for(size_t index = 0; index < f_dim/2; index++){
//        fprintf(p, "%+.6E %+.6E\n", ((float*)(eigenvectors[0][1]))[index*2], ((float*)(eigenvectors[0][1]))[index*2+1]);
//    }

	for(int i = 0; i < eigenvectors.size(); ++i){
		// Test the correctness of the eigenvectors and eigenvalues
#pragma omp parallel
{
		Fermion_t aa = lattice.bf.threadedAllocFermion();
		Fermion_t bb = lattice.bf.threadedAllocFermion();
		Fermion_t cc = lattice.bf.threadedAllocFermion();

		int me = lattice.bf.thread_barrier();
		lattice.bf.Mprec(eigenvectors[i][1], aa, cc, DaggerNo); // aa = M * eigenvector
		lattice.bf.Mprec(aa, bb, cc, DaggerYes);    // bb = Mdag * aa
		lattice.bf.axpby(cc, bb, eigenvectors[i][1], +1., -1.*eigenvalues[i]);
		std::complex<double> matrix_element = lattice.bf.inner(bb, eigenvectors[i][1]);
		std::complex<double> error = lattice.bf.inner(cc, cc);
		std::complex<double> norm = lattice.bf.inner(eigenvectors[i][1], eigenvectors[i][1]);
		if(not me){
			matrix_elements[i] = matrix_element.real();
		}
		if(lattice.bf.isBoss() && !me) {
	//		printf("ratio[0]aa = %.12e\n", ((float*)bb)[0]/((float*)eigenvectors[i][1])[0]);
	//		printf("ratio[2]aa = %.12e\n", ((float*)bb)[2]/((float*)eigenvectors[i][1])[2]);
	//		printf("ratio[4]aa = %.12e\n", ((float*)bb)[4]/((float*)eigenvectors[i][1])[4]);
			printf("#%04d: matrix_ele = %.12E+i%.12E; ", i, matrix_element.real(), matrix_element.imag());
			printf("error_norm = %.12E+i%.12E; ", error.real(), error.imag());
			printf("norm_eigen = %.12E+i%.12E; ", norm.real(), norm.imag());
			printf("eigenvalue = %.12E.\n", eigenvalues[i]);
		}	
		
		lattice.bf.threadedFreeFermion(aa);
		lattice.bf.threadedFreeFermion(bb);
		lattice.bf.threadedFreeFermion(cc);
}
	}
 	
	lattice.BondCond();  	
	GJP.Tbc(BND_CND_PRD); 
	// GJP.Xbc(BND_CND_PRD); 
	
	return 0;
}

void twist_tbc(GnoneFbfm& lattice, int t){
	size_t fix_t_blk_size = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites();
	size_t t_min, t_max;
	t_min = GJP.TnodeCoor()*GJP.TnodeSites();
	t_max = t_min + GJP.TnodeSites();

	VRB.Result("[twist_tbc]", "[twist_tbc]", "fix_t_blk_size = %d\n", fix_t_blk_size);

	long num_flip = 0;

	Matrix* gfp = lattice.GaugeField();
	if(t_min <= t and t < t_max){
// have the wanted links.
// #pragma omp parallel for
		for(size_t index = (t-t_min)*fix_t_blk_size; index < (t-t_min+1)*fix_t_blk_size; index++){
			gfp[index*4+3] *= -1.; // ONLY flip the links in t-direction
			num_flip++;
		}
	}

	qlat::glb_sum(num_flip);
	VRB.Result("[twist_tbc]", "[twist_tbc]", "num_flip = %d\n", num_flip);
	return;
}

void test_gt(void* cps_vct, GnoneFbfm& lattice, int t){
	size_t fix_t_blk_size = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites()*lattice.bf.Ls*3*4;
	size_t t_min, t_max;
	t_min = GJP.TnodeCoor()*GJP.TnodeSites();
	t_max = t_min + GJP.TnodeSites();

	VRB.Result("[test_gt]", "[test_gt]", "fix_t_blk_size = %d\n", fix_t_blk_size);
	
	ComplexF* vctp = (ComplexF*)cps_vct;
	for(size_t offset = 0; offset < GJP.TnodeSites()*fix_t_blk_size; offset++){
		vctp[offset] = +1.; // ONLY flip the links in t-direction
	}

	if(t_min <= t and t < t_max){
		for(size_t offset = (t-t_min)*fix_t_blk_size; offset < (t-t_min+1)*fix_t_blk_size; offset++){
			vctp[offset] = -1.;
		}
	}

	return;
}

void make_src(void* cps_vct, GnoneFbfm& lattice, int t){
	size_t fix_t_blk_size = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites()*lattice.bf.Ls*3*4;
	size_t t_min, t_max;
	t_min = GJP.TnodeCoor()*GJP.TnodeSites();
	t_max = t_min + GJP.TnodeSites();

	VRB.Result("[test_gt]", "[test_gt]", "fix_t_blk_size = %d\n", fix_t_blk_size);
	
	ComplexF* vctp = (ComplexF*)cps_vct;
	for(size_t offset = 0; offset < GJP.TnodeSites()*fix_t_blk_size; offset++){
		vctp[offset] = 0.; // ONLY flip the links in t-direction
	}

	if(t_min <= t and t < t_max){
		for(size_t offset = (t-t_min)*fix_t_blk_size; offset < (t-t_min+1)*fix_t_blk_size; offset++){
			vctp[offset] = 1.;
		}
	}

	return;
}

void apply_u1(void* x, void* bfm_vct, GnoneFbfm& lattice){
	size_t fix_t_blk_size = GJP.XnodeSites()*GJP.YnodeSites()*GJP.ZnodeSites()*lattice.bf.Ls*3*4/2; // number of ComplexF
	ComplexF* xp = (ComplexF*)x;
	ComplexF* bfmp = (ComplexF*)bfm_vct;
	for(size_t offset = 0; offset < GJP.TnodeSites()*fix_t_blk_size; offset++){
		xp[offset] *= bfmp[offset]; // ONLY flip the links in t-direction
	}
}

void test_local_deflation(GnoneFbfm& lattice){
	std::vector<qlat::U1GaugeTransform> u1gts;
	std::vector<qlat::FourInterval> global_partition;

	qlat::Coordinate local_size(GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites());
	qlat::Geometry geo; geo.init(qlat::get_geometry_node(), 1, local_size);
	geo.eo = 0;

	qlat::Printf("%s\n", show(geo).c_str());
	qlat::Coordinate tw_par(1,1,1,64);

	make_local_deflation_plan(u1gts, global_partition, geo, tw_par);
	VRB.Result("[LD]", "[LD]", "u1gts.size() = %d\n", u1gts.size());

// Test how U1 gauge tranformation applies on bfm_vct

	GJP.Tbc(BND_CND_PRD); 
	// GJP.Xbc(BND_CND_APRD); 
	lattice.ImportGauge(); // BondCond() is virtual so gauge field will imported to bfm

	lattice.bf.GeneralisedFiveDimEnd();
	lattice.bf.GeneralisedFiveDimInit();
	lattice.bf.comm_init();

	size_t bfm_vct_fdim = GJP.VolNodeSites()*lattice.FsiteSize()/2; // FsiteSize = 2 * 3 * 4. The last 2 for even-odd preconditioning
	size_t cps_vct_fdim = GJP.VolNodeSites()*lattice.FsiteSize(); // FsiteSize = 2 * 3 * 4. The last 2 for even-odd preconditioning
	VRB.Result("[LD]", "[LD]", "bfm_vct_fdim = %d\n", bfm_vct_fdim);
	VRB.Result("[LD]", "[LD]", "cps_vct_fdim = %d\n", cps_vct_fdim);

	void* cps_vct = new char[cps_vct_fdim*sizeof(float)];
	void* cps_src1 = new char[cps_vct_fdim*sizeof(float)];
	void* cps_src2 = new char[cps_vct_fdim*sizeof(float)];
	test_gt(cps_vct, lattice, 30);
	make_src(cps_src1, lattice, 32);
	make_src(cps_src2, lattice, 25);
	Fermion_t bfm_vct[2] = { lattice.bf.allocFermion(), lattice.bf.allocFermion() };
	Fermion_t bfm_src1[2] = { lattice.bf.allocFermion(), lattice.bf.allocFermion() };
	Fermion_t bfm_src2[2] = { lattice.bf.allocFermion(), lattice.bf.allocFermion() };
	lattice.bf.cps_impexFermion_s((float*)cps_vct, bfm_vct, 1);
	lattice.bf.cps_impexFermion_s((float*)cps_src1, bfm_src1, 1);
	lattice.bf.cps_impexFermion_s((float*)cps_src2, bfm_src2, 1);

	Fermion_t x = lattice.bf.allocFermion();
	Fermion_t y = lattice.bf.allocFermion();
	Fermion_t yg = lattice.bf.allocFermion();
	Fermion_t t1 = lattice.bf.allocFermion();
	Fermion_t t2 = lattice.bf.allocFermion();
	Fermion_t par_vct = lattice.bf.allocFermion();
	Fermion_t psv_vct = lattice.bf.allocFermion();
	
	Fermion_t xb = lattice.bf.allocFermion();
	Fermion_t sr = lattice.bf.allocFermion();
	Fermion_t ei = lattice.bf.allocFermion();

	multi1d<Fermion_t[2]> eigenvectors(2);
	multi1d<float> eigenvalues(2);
//
	eigenvectors[0][1] = bfm_src1[1];
	eigenvectors[1][1] = bfm_src2[1];
	eigenvalues[0] = 1E5;
	eigenvalues[1] = 2E5;

#pragma omp parallel for
	for(size_t index = 0; index < bfm_vct_fdim; index++){
		((float*)x)[index] = 1.;
	}

#pragma omp parallel for
	for(size_t index = 0; index < bfm_vct_fdim; index++){
		((float*)sr)[index] = 1.;
	}
#pragma omp parallel for
	for(size_t index = 0; index < bfm_vct_fdim; index++){
		((float*)ei)[index] = 1.;
	}
#pragma omp parallel for
	for(size_t index = 0; index < bfm_vct_fdim; index++){
		((float*)xb)[index] = 0.;
	}

#pragma omp parallel
{
	Fermion_t yp = lattice.bf.threadedAllocFermion();
	Fermion_t xp = lattice.bf.threadedAllocFermion();
	for(size_t p = 0; p < u1gts.size(); p++){
		extract_par_vct_from_bfm_vct(yp, bfm_src1[1], lattice.bf.Ls, global_partition[p], geo);
		apply_u1_gauge_tranform_on_bfm_vct(yp, lattice.bf.Ls, u1gts[p]);
		lattice.bf.deflate(xp, yp, &eigenvectors, &eigenvalues, 1);
		apply_u1_gauge_tranform_on_bfm_vct(xp, lattice.bf.Ls, u1gts[p]);
		lattice.bf.axpby(xb, xp, xb, 1., 1.);
	}	
	lattice.bf.threadedFreeFermion(yp);
	lattice.bf.threadedFreeFermion(xp);
}

	extract_par_vct_from_bfm_vct(sr, sr, lattice.bf.Ls, global_partition[12], geo);

#pragma omp parallel
{
	local_deflate(lattice.bf, xb, x, &eigenvectors, &eigenvalues, u1gts, global_partition);
}

	std::complex<double> xbxb;
	std::complex<double> bfm_src_sqr;
#pragma omp parallel
{
	xbxb = lattice.bf.inner(xb, xb);
	bfm_src_sqr = lattice.bf.inner(bfm_src1[1], bfm_src1[1]);
}
	VRB.Result("[LD]", "[LD]", "(xb     , xb     ) = %.8E\n", xbxb.real());
	VRB.Result("[LD]", "[LD]", "(bfm_src, bfm_src) = %.8E\n", bfm_src_sqr.real());

#pragma omp parallel
{
	local_deflate_fft(lattice.bf, xb, x, &eigenvectors, &eigenvalues, u1gts, global_partition, tw_par);
}

#pragma omp parallel
{
	xbxb = lattice.bf.inner(xb, xb);
	bfm_src_sqr = lattice.bf.inner(bfm_src1[1], bfm_src1[1]);
}
	VRB.Result("[LD]", "[LD]", "(xb     , xb     ) = %.8E\n", xbxb.real());
	VRB.Result("[LD]", "[LD]", "(bfm_src, bfm_src) = %.8E\n", bfm_src_sqr.real());


	std::complex<double> xx;
#pragma omp parallel
{
	xx = lattice.bf.inner(x, x);
}
	VRB.Result("[LD]", "[LD]", "(x      , x      ) = %.8E\n", xx.real());
	
	extract_par_vct_from_bfm_vct(par_vct, x, lattice.bf.Ls, global_partition[13], geo);
	extract_par_vct_from_bfm_vct(psv_vct, x, lattice.bf.Ls, global_partition[13], geo);
	std::complex<double> xxp;
#pragma omp parallel
{
	xxp = lattice.bf.inner(par_vct, psv_vct);
}
	VRB.Result("[LD]", "[LD]", "(par_vct, psv_vct) = %.8E\n", xxp.real());

	apply_u1_gauge_tranform_on_bfm_vct(par_vct, lattice.bf.Ls, u1gts[13]);
//	apply_u1(par_vct, bfm_vct[1], lattice);
#pragma omp parallel
{
	xxp = lattice.bf.inner(par_vct, psv_vct);
}
	VRB.Result("[LD]", "[LD]", "(par_vct, psv_vct) = %.8E\n", xxp.real());

#pragma omp parallel
{
	lattice.bf.Mprec(x, t1, t2, DaggerNo); // t1 = M * x
	lattice.bf.Mprec(t1, y, t2, DaggerYes);    // y = Mdag * t1
}
//	lattice.BondCond(); // BondCond() is virtual so gauge field will imported to bfm
	
	// measure_plaq(lattice);

	twist_tbc(lattice, 62);
	twist_tbc(lattice, 63);
	lattice.ImportGauge(); // BondCond() is virtual so gauge field will imported to bfm
	
	// measure_plaq(lattice);

//	Somehow after f_bfm.BondCond() we need to do all these again, otherwise communication will be problematic.
	lattice.bf.GeneralisedFiveDimEnd();
	lattice.bf.GeneralisedFiveDimInit();
	lattice.bf.comm_init();

	std::complex<double> inner;
#pragma omp parallel
{
	inner = lattice.bf.inner(x, y);
}
	VRB.Result("[LD]", "[LD]", "inner = %.8E + i %.8E\n", inner.real(), inner.imag());
	apply_u1_gauge_tranform_on_bfm_vct(x, lattice.bf.Ls, u1gts[13]);
	apply_u1_gauge_tranform_on_bfm_vct(y, lattice.bf.Ls, u1gts[13]);
//	apply_u1(x, bfm_vct[1], lattice);
//	apply_u1(y, bfm_vct[1], lattice);

#pragma omp parallel
{
	inner = lattice.bf.inner(x, y);
}	
	VRB.Result("[LD]", "[LD]", "inner = %.8E + i %.8E\n", inner.real(), inner.imag());
	
	float diff, zero, test;

#pragma omp parallel
{
	lattice.bf.Mprec(x, t1, t2, DaggerNo); // t1 = M * x
	lattice.bf.Mprec(t1, yg, t2, DaggerYes);    // y = Mdag * t1
	diff = lattice.bf.axpy_norm(t1, yg, y, -1.);
	zero = lattice.bf.axpy_norm(t1, y, y, -1.);
	test = lattice.bf.axpy_norm(t2, y, y, 0.);
}
	VRB.Result("[LD]", "[LD]", "testA= %.8E\n", test);
	apply_u1_gauge_tranform_on_bfm_vct(y, lattice.bf.Ls, u1gts[13]);
#pragma omp parallel
{
	test = lattice.bf.axpy_norm(t2, y, y, 0.);
}
	VRB.Result("[LD]", "[LD]", "diff = %.8E\n", diff);
	VRB.Result("[LD]", "[LD]", "zero = %.8E\n", zero);
	VRB.Result("[LD]", "[LD]", "testB= %.8E\n", test);

	lattice.bf.freeFermion(x);
	lattice.bf.freeFermion(y);
	lattice.bf.freeFermion(yg);
	lattice.bf.freeFermion(t1);
	lattice.bf.freeFermion(t2);
	lattice.bf.freeFermion(par_vct);
	lattice.bf.freeFermion(psv_vct);
	
	lattice.bf.freeFermion(xb);
	lattice.bf.freeFermion(sr);
	lattice.bf.freeFermion(ei);

	delete [] cps_vct;
	delete [] cps_src1;

	lattice.BondCond();  	
	GJP.Tbc(BND_CND_PRD); 
}

/*
long low_mode_submatrix(GnoneFbfm& lattice,
	multi1d<Fermion_t[2]>& eigenvectors, multi1d<float>& eigenvalues)
{
//	lattice.ImportGauge();
	// GJP.Tbc(BND_CND_APRD); 
	GJP.Tbc(BND_CND_PRD); 
	// GJP.Xbc(BND_CND_APRD); 
	lattice.BondCond(); // BondCond() is virtual so gauge field will imported to bfm

	lattice.bf.GeneralisedFiveDimEnd();
	lattice.bf.GeneralisedFiveDimInit();
	lattice.bf.comm_init();

    lattice.SetBfmArg(lqpropw_arg.cg.mass);

//	char output_file[1024];
//	sprintf(output_file, "Chritoph.%03d", UniqueID());
//	FILE* p = fopen(output_file, "w");
//    size_t f_dim = GJP.VolNodeSites () * lattice.FsiteSize () / 2;
//    for(size_t index = 0; index < f_dim/2; index++){
//        fprintf(p, "%+.6E %+.6E\n", ((float*)(eigenvectors[0][1]))[index*2], ((float*)(eigenvectors[0][1]))[index*2+1]);
//    }

	FILE* p;
	if(not UniqueID()){
		p = fopen("subMatrix_2000_TbcPRD_light.dat", "w");
	}
	
	int sub_matrix_size = eigenvectors.size();

	qlat::Matrix<sub_matrix_size> sub_matrix; set_zero(sub_matrix);

#pragma omp parallel
{
	Fermion_t aa = lattice.bf.threadedAllocFermion();
	Fermion_t bb = lattice.bf.threadedAllocFermion();
	Fermion_t cc = lattice.bf.threadedAllocFermion();

	for(int i = 0; i < 2000; ++i){
	for(int j = 0; j < i+1; ++j){

		int me = lattice.bf.thread_barrier();
		lattice.bf.Mprec(eigenvectors[i][1], aa, cc, DaggerNo); // aa = M * eigenvector
		lattice.bf.Mprec(aa, bb, cc, DaggerYes);    // bb = Mdag * aa
		std::complex<double> matrix_element = lattice.bf.inner(bb, eigenvectors[j][1]);
		if(lattice.bf.isBoss() && !me) {
			printf("Now caculating element (%03d,%03d) .\n", i, j);
			// fprintf(p, "%18.12E %18.12E\n", matrix_element.real(), matrix_element.imag());
			sub_matrix[i][j] = matrix_element;
			sub_matrix[j][i] = conj(matrix_element);
		}	
		
	}}
	
	lattice.bf.threadedFreeFermion(aa);
	lattice.bf.threadedFreeFermion(bb);
	lattice.bf.threadedFreeFermion(cc);

	

	bf.caxpy()

	Eigen::SelfAdjointEigenSolver<decltype(sub_matrix)> es(sub_matrix);
	for(int i = 0; i < eigenvectors.size(); i++){
		eigenvalues[i] = es.eigenvalues()[i];
			
	}


}

	// TODO: Do we have the problem os getting defferent values on defferent nodes?
	
	



	lattice.BondCond();  	
	GJP.Tbc(BND_CND_PRD); 
	// GJP.Xbc(BND_CND_PRD); 
	
	return 0;
}

*/

// comp_read_eigenvectors() function in cps_pp/work
void load_eigenvectors(GnoneFbfm& lattice, char* directory, 
		multi1d<Fermion_t[2]>& eigenvectors, multi1d<float>& eigenvalues)
{
	char *fname = "load_eigenvectors(lattice&, int)";
	VRB.Func("", fname);
	
	Float etime = time_elapse ();
	
	char cache_name[1024];
	snprintf (cache_name, 1024, "cache_0_mass%g", lanczos_arg.mass);
	
	static EigenCache ecache(cache_name); // only have one instance

	int fsize = GJP.VolNodeSites () * lattice.FsiteSize () / 2 / 2;	// first 2 for even-odd preconditioning, second 2 for single precesion

	int number_of_eigenvectors_to_read = eigenvalues.size();

	ecache.alloc(directory, number_of_eigenvectors_to_read, fsize);
		
	const int n_fields = GJP.SnodeSites ();
	const int f_size_per_site = lattice.FsiteSize () / n_fields / 2;
	EigenContainer eigcon(lattice, directory, number_of_eigenvectors_to_read, f_size_per_site / 2, n_fields, &ecache);
		
	eigcon.load_rbc(directory, number_of_eigenvectors_to_read);
	
	for(int i = 0; i < number_of_eigenvectors_to_read; i++){
		eigenvectors[i][1] = (Fermion_t)ecache.vec_ptr(i);
		eigenvalues[i] = sqrt(ecache.eval_address()[i]);
		
		float *temp = (float*)eigenvectors[i][1];
		double sum = 0.;
		
		for (size_t ind = 0; ind < (size_t) fsize * ( sizeof(double)/sizeof(float) ); ind++){
			sum += temp[ind] * temp[ind];
		}

      	glb_sum (&sum);
      
		if(!UniqueID()) printf("eigenvalue number %d = %.12e\n", i, eigenvalues[i]);
		if(!UniqueID()) printf("eigenvector number %d norm = %.12e\n", i, sum);
	}
	
	etime = time_elapse ();
	if(!UniqueID()) printf("Time for Lanczos %g\n", etime);

	VRB.FuncEnd("",fname);
}

void run_contractions(const AllProp &sprop, const AllProp &stwst,
		const AllProp &lprop, const AllProp &ltwst,
		const string &rdir,
		int traj, PROP_TYPE ptype)
{
	const char *fname = "run_contractions()";
	VRB.Result(cname, fname, "run_contractions for trajectory %d with prop type %d\n", traj, ptype);

	const string trajs = string(".") + tostring(traj);

	//////////////////////////////////////////////////////////////////////
	// 1. meson contractions

	//Greg: Can skip contractions involving twists, so I've commented them out.

	// pion and kaon
	VRB.Result(cname, fname, "Doing untwisted point sink pion contractions\n");
	run_meson_pt(lprop, lprop, GAMMA_5, GAMMA_5, rdir + "/pion-00WP" + trajs, ptype);
//	run_meson_pt(lprop, ltwst, GAMMA_5, GAMMA_5, rdir + "/pion-01WP" + trajs, ptype);
//	run_meson_pt(ltwst, lprop, GAMMA_5, GAMMA_5, rdir + "/pion-10WP" + trajs, ptype);
//	run_meson_pt(ltwst, ltwst, GAMMA_5, GAMMA_5, rdir + "/pion-11WP" + trajs, ptype);
	
	VRB.Result(cname, fname, "Doing untwisted point sink kaon contractions\n");
	run_meson_pt(sprop, lprop, GAMMA_5, GAMMA_5, rdir + "/kaon-00WP" + trajs, ptype);
//	run_meson_pt(stwst, lprop, GAMMA_5, GAMMA_5, rdir + "/kaon-10WP" + trajs, ptype);
//	run_meson_pt(sprop, ltwst, GAMMA_5, GAMMA_5, rdir + "/kaon-01WP" + trajs, ptype);
//	run_meson_pt(stwst, ltwst, GAMMA_5, GAMMA_5, rdir + "/kaon-11WP" + trajs, ptype);

	VRB.Result(cname, fname, "Doing untwisted wall sink pion contractions\n");
	run_meson_wall(lprop, lprop, GAMMA_5, GAMMA_5, rdir + "/pion-00WW" + trajs, ptype);
//	run_meson_wall(lprop, ltwst, GAMMA_5, GAMMA_5, rdir + "/pion-01WW" + trajs, ptype);
//	run_meson_wall(ltwst, lprop, GAMMA_5, GAMMA_5, rdir + "/pion-10WW" + trajs, ptype);
//	run_meson_wall(ltwst, ltwst, GAMMA_5, GAMMA_5, rdir + "/pion-11WW" + trajs, ptype);
	
	VRB.Result(cname, fname, "Doing untwisted wall sink kaon contractions\n");
	run_meson_wall(sprop, lprop, GAMMA_5, GAMMA_5, rdir + "/kaon-00WW" + trajs, ptype);
//	run_meson_wall(stwst, lprop, GAMMA_5, GAMMA_5, rdir + "/kaon-10WW" + trajs, ptype);
//	run_meson_wall(sprop, ltwst, GAMMA_5, GAMMA_5, rdir + "/kaon-01WW" + trajs, ptype);
//	run_meson_wall(stwst, ltwst, GAMMA_5, GAMMA_5, rdir + "/kaon-11WW" + trajs, ptype);

	// scalar meson (sigma) contractions.
	VRB.Result(cname, fname, "Doing untwisted light-light scalar contractions\n");
	run_meson_wall(lprop, lprop, ID,      ID,      rdir + "/sigma-00WW" + trajs, ptype);
	run_meson_disc(lprop, lprop, ID,      ID,      rdir + "/sigma-dis-00WW" + trajs, ptype);

	// eta eta' contractions
	//
	// We share the light-light propagator with pion contractions.
	VRB.Result(cname, fname, "Doing untwisted strange-strange pseudoscalar contractions\n");
	run_meson_wall(sprop, sprop, GAMMA_5, GAMMA_5, rdir + "/ss-00WW" + trajs, ptype);
	VRB.Result(cname, fname, "Doing untwisted light-strange pseudoscalar disconnected contractions\n");
	run_meson_disc(lprop, sprop, GAMMA_5, GAMMA_5, rdir + "/ls-dis-00WW" + trajs, ptype);

	// f_K and f_pi measurements
	VRB.Result(cname, fname, "Doing a bunch of local axial current contractions for f_K and f_pi\n");
	run_meson_pt(lprop, lprop, GAMMA_35, GAMMA_5, rdir + "/fp-00WP" + trajs, ptype);
//	run_meson_pt(lprop, ltwst, GAMMA_35, GAMMA_5, rdir + "/fp-01WP" + trajs, ptype);
//	run_meson_pt(ltwst, lprop, GAMMA_35, GAMMA_5, rdir + "/fp-10WP" + trajs, ptype);
//	run_meson_pt(ltwst, ltwst, GAMMA_35, GAMMA_5, rdir + "/fp-11WP" + trajs, ptype);
	
	run_meson_pt(sprop, lprop, GAMMA_35, GAMMA_5, rdir + "/fk-00WP" + trajs, ptype);
//	run_meson_pt(sprop, ltwst, GAMMA_35, GAMMA_5, rdir + "/fk-01WP" + trajs, ptype);
//	run_meson_pt(stwst, lprop, GAMMA_35, GAMMA_5, rdir + "/fk-10WP" + trajs, ptype);
//	run_meson_pt(stwst, ltwst, GAMMA_35, GAMMA_5, rdir + "/fk-11WP" + trajs, ptype);

	run_meson_pt(lprop, lprop, GAMMA_5, GAMMA_35, rdir + "/fpr-00WP" + trajs, ptype);
//	run_meson_pt(lprop, ltwst, GAMMA_5, GAMMA_35, rdir + "/fpr-01WP" + trajs, ptype);
//	run_meson_pt(ltwst, lprop, GAMMA_5, GAMMA_35, rdir + "/fpr-10WP" + trajs, ptype);
//	run_meson_pt(ltwst, ltwst, GAMMA_5, GAMMA_35, rdir + "/fpr-11WP" + trajs, ptype);
	
	run_meson_pt(sprop, lprop, GAMMA_5, GAMMA_35, rdir + "/fkr-00WP" + trajs, ptype);
//	run_meson_pt(sprop, ltwst, GAMMA_5, GAMMA_35, rdir + "/fkr-01WP" + trajs, ptype);
//	run_meson_pt(stwst, lprop, GAMMA_5, GAMMA_35, rdir + "/fkr-10WP" + trajs, ptype);
//	run_meson_pt(stwst, ltwst, GAMMA_5, GAMMA_35, rdir + "/fkr-11WP" + trajs, ptype);

	run_meson_pt(lprop, lprop, GAMMA_35, GAMMA_35, rdir + "/ap-00WP" + trajs, ptype);
//	run_meson_pt(lprop, ltwst, GAMMA_35, GAMMA_35, rdir + "/ap-01WP" + trajs, ptype);
//	run_meson_pt(ltwst, lprop, GAMMA_35, GAMMA_35, rdir + "/ap-10WP" + trajs, ptype);
//	run_meson_pt(ltwst, ltwst, GAMMA_35, GAMMA_35, rdir + "/ap-11WP" + trajs, ptype);
	
	run_meson_pt(sprop, lprop, GAMMA_35, GAMMA_35, rdir + "/ak-00WP" + trajs, ptype);
//	run_meson_pt(sprop, ltwst, GAMMA_35, GAMMA_35, rdir + "/ak-01WP" + trajs, ptype);
//	run_meson_pt(stwst, lprop, GAMMA_35, GAMMA_35, rdir + "/ak-10WP" + trajs, ptype);
//	run_meson_pt(stwst, ltwst, GAMMA_35, GAMMA_35, rdir + "/ak-11WP" + trajs, ptype);

	run_meson_wall(lprop, lprop, GAMMA_35, GAMMA_5, rdir + "/fp-00WW" + trajs, ptype);
//	run_meson_wall(lprop, ltwst, GAMMA_35, GAMMA_5, rdir + "/fp-01WW" + trajs, ptype);
//	run_meson_wall(ltwst, lprop, GAMMA_35, GAMMA_5, rdir + "/fp-10WW" + trajs, ptype);
//	run_meson_wall(ltwst, ltwst, GAMMA_35, GAMMA_5, rdir + "/fp-11WW" + trajs, ptype);
	
	run_meson_wall(sprop, lprop, GAMMA_35, GAMMA_5, rdir + "/fk-00WW" + trajs, ptype);
//	run_meson_wall(sprop, ltwst, GAMMA_35, GAMMA_5, rdir + "/fk-01WW" + trajs, ptype);
//	run_meson_wall(stwst, lprop, GAMMA_35, GAMMA_5, rdir + "/fk-10WW" + trajs, ptype);
//	run_meson_wall(stwst, ltwst, GAMMA_35, GAMMA_5, rdir + "/fk-11WW" + trajs, ptype);

	// rho meson
	VRB.Result(cname, fname, "Doing a bunch of vector meson (rho) contractions\n");
	run_meson_pt(lprop, lprop, GAMMA_0, GAMMA_0, rdir + "/rho-x-00WP" + trajs, ptype);
	run_meson_pt(lprop, lprop, GAMMA_1, GAMMA_1, rdir + "/rho-y-00WP" + trajs, ptype);
	run_meson_pt(lprop, lprop, GAMMA_2, GAMMA_2, rdir + "/rho-z-00WP" + trajs, ptype);
	run_meson_wall(lprop, lprop, GAMMA_0, GAMMA_0, rdir + "/rho-x-00WW" + trajs, ptype);
	run_meson_wall(lprop, lprop, GAMMA_1, GAMMA_1, rdir + "/rho-y-00WW" + trajs, ptype);
	run_meson_wall(lprop, lprop, GAMMA_2, GAMMA_2, rdir + "/rho-z-00WW" + trajs, ptype);

	//////////////////////////////////////////////////////////////////////
	// 2. Omega baryon
	VRB.Result(cname, fname, "Doing a bunch of omega baryon contractions\n");
	run_omega_pt(sprop, GAMMA_0, rdir + "/sss-x-00WP" + trajs, ptype);
	run_omega_pt(sprop, GAMMA_1, rdir + "/sss-y-00WP" + trajs, ptype);
	run_omega_pt(sprop, GAMMA_2, rdir + "/sss-z-00WP" + trajs, ptype);
	run_omega_pt(sprop, GAMMA_3, rdir + "/sss-t-00WP" + trajs, ptype);
	run_omega_pt(sprop, GAMMA_5, rdir + "/sss-5-00WP" + trajs, ptype);

	//////////////////////////////////////////////////////////////////////
	// 3. Kl3

	run_kl3(sprop, lprop, lprop, rdir + "/kl3-00" + trajs, ptype);
//	run_kl3(sprop, lprop, ltwst, rdir + "/kl3-01" + trajs, ptype);
//	run_kl3(stwst, lprop, lprop, rdir + "/kl3-10" + trajs, ptype);
//	run_kl3(stwst, ltwst, lprop, rdir + "/kl3-11" + trajs, ptype);

	// The following contractions are used for Z_V.
	// Useful for the double ratio method or the UKQCD method.
	run_kl3(lprop, lprop, lprop, rdir + "/zpa-00" + trajs, ptype);
	run_kl3(sprop, lprop, sprop, rdir + "/zka-00" + trajs, ptype);
	run_kl3(lprop, sprop, lprop, rdir + "/zkb-00" + trajs, ptype);
	run_kl3(sprop, sprop, sprop, rdir + "/zss-00" + trajs, ptype);

	// Since line 1 and 3 both carry momentum, are their directions
	// consistent?
	// I think they are.
//	run_kl3(ltwst, lprop, ltwst, rdir + "/zpa-11" + trajs, ptype);
//	run_kl3(lprop, ltwst, lprop, rdir + "/zpb-11" + trajs, ptype);
//	run_kl3(stwst, lprop, stwst, rdir + "/zka-11" + trajs, ptype);
//	run_kl3(lprop, stwst, lprop, rdir + "/zkb-11" + trajs, ptype);

	//////////////////////////////////////////////////////////////////////
	// 4. Bk
	VRB.Result(cname, fname, "Doing B_k contractions\n");
	run_bk(lprop, sprop, lprop, sprop, rdir + "/bk" + trajs, ptype);
}

void run_omega_box_contractions(const AllProp &sprop_box,
		const string &rdir,
		int traj, PROP_TYPE ptype)
{
	const char *fname = "run_contractions()";

	const string trajs = string(".") + tostring(traj);

	//////////////////////////////////////////////////////////////////////
	// 2. meson contractions

	// eta eta' contractions
	//
	// We share the light-light propagator with pion contractions.
	run_meson_pt  (sprop_box, sprop_box, GAMMA_5, GAMMA_5, rdir + "/box-ss-00WP" + trajs, ptype);
	run_meson_wall(sprop_box, sprop_box, GAMMA_5, GAMMA_5, rdir + "/box-ss-00WW" + trajs, ptype);

	//////////////////////////////////////////////////////////////////////
	// 2. Omega baryon
	run_omega_pt(sprop_box, GAMMA_0, rdir + "/box-sss-x-00WP" + trajs, ptype);
	run_omega_pt(sprop_box, GAMMA_1, rdir + "/box-sss-y-00WP" + trajs, ptype);
	run_omega_pt(sprop_box, GAMMA_2, rdir + "/box-sss-z-00WP" + trajs, ptype);
	run_omega_pt(sprop_box, GAMMA_3, rdir + "/box-sss-t-00WP" + trajs, ptype);
	run_omega_pt(sprop_box, GAMMA_5, rdir + "/box-sss-5-00WP" + trajs, ptype);
}


void run_k2pipi_contractions(const AllProp &sprop,
		const AllProp &uprop,
		const AllProp &dprop,
		const string &rdir,
		int traj, const int mom[3],
		PROP_TYPE ptype)
{
	const string trajs = string(".") + tostring(traj);

	// zero momentum pi pi scattering
	int zmom[3] = {0, 0, 0};
	run_2pionDC(uprop, uprop, rdir + "/2pion000" + trajs, ptype, zmom);
/*
	// Only need the zero-momentum case to compute the S-wave pi-pi scattering length
	for(unsigned i = 0; i < 8; ++i) {
		int p[3];
		p[0] = (i & 1) ? -mom[0] : mom[0];
		p[1] = (i & 2) ? -mom[1] : mom[1];
		p[2] = (i & 4) ? -mom[2] : mom[2];
	
		const string fn = rdir + "/2pion"
		+ tostring(p[0]) + tostring(p[1]) + tostring(p[2])
		+ trajs;
	
		run_2pionDC(uprop, dprop, fn, ptype, p);
        
		const string tw_pion_fn = rdir + "/pion-00WW-HP"
        + tostring(p[0]) + tostring(p[1]) + tostring(p[2])
        + trajs;

        run_meson_wall(uprop, dprop, GAMMA_5, GAMMA_5, tw_pion_fn, ptype, p);
	}
*/
	// Added by Jackson
	// Somewhat confusing phase factor ... Need to make sure this is right.
	// run_meson_wall(uprop, dprop, GAMMA_5, GAMMA_5, rdir + "/pion-00WW-HP" + trajs, ptype, mom);

	// K->pipi without momentum.
	// run_k2pipi(sprop, uprop, uprop, rdir + "/k2pipi-0" + trajs, ptype);

	// K->pipi with momentum.
	// run_k2pipi(sprop, uprop, dprop, rdir + "/k2pipi-1" + trajs, ptype);
}

class FixGauge
{
	public:
		FixGauge(cps::Lattice &lat,
				cps::FixGaugeArg &fix_gauge_arg,
				int traj, const char* gf_matrices = NULL)
		{
			char buf[256];
			sprintf(buf, "../results/fg-bc.%d", traj);
			cps::Fclose(cps::Fopen(buf, "w"));
			com_fg.set_filename(buf);

			fg = new cps::AlgFixGauge(lat, &com_fg, &fix_gauge_arg);
			VRB.Result("FixGauge", "FixGauge", "Doing gauge fixing for trajectory %d\n", traj);
			
			if(gf_matrices){
				VRB.Result("FixGauge", "FixGauge", "Reading gauge fixing matrices from %s\n", gf_matrices);
				qlat::Coordinate local_size(GJP.XnodeSites(), GJP.YnodeSites(), GJP.ZnodeSites(), GJP.TnodeSites());
				qlat::Geometry gf_geo; gf_geo.init(qlat::get_geometry_node(), 1, local_size); // multiplicity = 1 
				qlat::Field<cps::Matrix> gff; gff.init(gf_geo);
				qlat::serial_read_field(gff, std::string(gf_matrices), -qlat::get_data_size(gff) * qlat::get_num_node(), SEEK_END);
				VRB.Result("FixGauge", "FixGauge", "computed checksum = %08x\n", qlat::fieldChecksumSum32(gff));	
				std::printf("%.8f\n", gff.field[0].norm());
				
				fg->run(gff.field.data());
			}else{
				fg->run();
			}
		}

		~FixGauge() {
			fg->free();
			delete fg;
		}
	private:
		cps::CommonArg com_fg;
		cps::AlgFixGauge *fg;
};

// stw: twisting angle of the strange quark (connecting the operator
// and the kaon).
//
// ltw: twisting angle of the light quark (connecting the operator and
// the pion).
void run_all(Fbfm &lat,
		const double stw[4], // strange quark twists, for Kl3
		const double ltw[4], //   light quark twists, for Kl3
		const int mom[3],    // momentum for the d quark, used by K2pipi
		int traj, const char* gf_matrices_directory = NULL)
{
	const char *fname = "run_all()";
	VRB.Result(cname, fname, "Starting run_all for trajectory %d\n", traj);

	//////////////////////////////////////////////////////////////////////
	// 1. props: strange, strange twisted, light, light twisted

	// exact strange
	AllProp sprop(AllProp::DOUBLE), stwst(AllProp::DOUBLE);
	// exact light
	AllProp lprop_e(AllProp::DOUBLE), ltwst_e(AllProp::DOUBLE);
	// sloppy light, single precision
	AllProp lprop(AllProp::SINGLE), ltwst(AllProp::SINGLE);
	// exact strange, Z3 box source
	AllProp sprop_box(AllProp::DOUBLE);
	
	// TODO: For test only!!!
	// AllProp lprop_box(AllProp::DOUBLE);

	Float dtime0 = dclock();

	//this constructor does the gauge fixing, and the destructor
	//automatically frees the gauge fixing matrices at the end of run_all
	FixGauge fg(lat, fix_gauge_arg, traj, gf_matrices_directory);

	Float dtime01 = dclock();

	VRB.Result(cname, fname, "Starting Z3 box source inversions for trajectory %d\n", traj);
	// run_wall_box_prop(nullptr, &sprop, &sprop_box, s_ut_loc, lat, sqpropw_arg, box_arg, &s_eigcg_arg, traj, true);
	// run_box_prop_test_twist_all(&lprop_box, lat, lqpropw_arg, box_arg_3d, &l_eigcg_arg, traj);

	Float dtime02 = dclock();

	VRB.Result(cname, fname, "Doing Z3 box contractions for trajectory %d\n", traj);
	// run_omega_box_contractions(sprop_box, "../resultsPA", traj, PROP_PA);
	// run_omega_box_contractions(sprop_box, "../resultsP", traj, PROP_P);
	// run_omega_box_contractions(sprop_box, "../resultsA", traj, PROP_A);

	Float dtime1 = dclock();

	VRB.Result(cname, fname, "Starting propagator inversions for trajectory %d\n", traj);

	// l untwisted
	VRB.Result(cname, fname, "Doing light untwisted wall source propagators for trajectory %d\n", traj);
	run_wall_prop(&lprop_e, &lprop, l_ut_loc, lat, lqpropw_arg, &l_eigcg_arg, traj, true );

	// l twisted
	// twisted_bc(lat, ltw, true);
	// run_wall_prop(&ltwst_e, &ltwst, l_tw_loc, lat, lqpropw_arg, &l_eigcg_arg, traj, false);
	// twisted_bc(lat, ltw, false);

	// lat.unset_deflation();
	// s untwisted
	VRB.Result(cname, fname, "Doing strange untwisted wall source propagators for trajectory %d\n", traj);
	// run_wall_prop(NULL, &sprop, s_ut_loc, lat, sqpropw_arg, NULL, traj, true );
	run_wall_prop(nullptr, &sprop, s_ut_loc, lat, sqpropw_arg, &s_eigcg_arg, traj, true);

	// s twisted
	// twisted_bc(lat, stw, true);
	// run_wall_prop(NULL, &stwst, s_tw_loc, lat, sqpropw_arg, NULL, traj, false);
	// twisted_bc(lat, stw, false);

	Float dtime2 = dclock();

	VRB.Result(cname, fname, "Starting contractions for trajectory %d\n", traj);

	VRB.Result(cname, fname, "Doing contractions with exact light propagators\n");
	//run_contractions(sprop, stwst, lprop_e, ltwst_e, "../resultsEPA", traj, PROP_PA);
	//run_contractions(sprop, stwst, lprop_e, ltwst_e, "../resultsEP",  traj, PROP_P);
	run_contractions(sprop, stwst, lprop_e, ltwst_e, "../resultsEA",  traj, PROP_A);

	VRB.Result(cname, fname, "Doing contractions with inexact light propagators\n");
	//run_contractions(sprop, stwst, lprop, ltwst, "../resultsPA", traj, PROP_PA);
	//run_contractions(sprop, stwst, lprop, ltwst, "../resultsP",  traj, PROP_P);
	run_contractions(sprop, stwst, lprop, ltwst, "../resultsA",  traj, PROP_A);

	Float dtime3 = dclock();

	////////////////////////////////////////////////////////////////////////
	// I=2 K to pi pi
	// free unwanted propagators to save some memory.
	ltwst_e.clear();
	ltwst.clear();
	stwst.clear();

	//Greg: Don't need to run K to pi pi stuff, so I've them out

	// twisted light for K -> pi pi
	VRB.Result(cname, fname, "Doing K->pipi light twisted wall source propagators for trajectory %d\n", traj);
  // run_mom_prop(&ltwst_e, &ltwst, l_tw_loc, lat, lqpropw_arg, &l_eigcg_arg, traj, mom);

	Float dtime4 = dclock();

	VRB.Result(cname, fname, "Doing exact K->pipi contractions for trajectory %d\n", traj);
	// run_k2pipi_contractions(sprop, lprop_e, ltwst_e, "../resultsEPA", traj, mom, PROP_PA);
	// run_k2pipi_contractions(sprop, lprop_e, ltwst_e, "../resultsEP",  traj, mom, PROP_P);
	run_k2pipi_contractions(sprop, lprop_e, ltwst_e, "../resultsEA",  traj, mom, PROP_A);

	VRB.Result(cname, fname, "Doing inexact K->pipi contractions for trajectory %d\n", traj);
	// run_k2pipi_contractions(sprop, lprop, ltwst, "../resultsPA", traj, mom, PROP_PA);
	// run_k2pipi_contractions(sprop, lprop, ltwst, "../resultsP",  traj, mom, PROP_P);
  run_k2pipi_contractions(sprop, lprop, ltwst, "../resultsA",  traj, mom, PROP_A);

	Float dtime5 = dclock();

	VRB.Result(cname, fname, "fix gauge    = %4.2e seconds\n", dtime01 - dtime0);
	VRB.Result(cname, fname, "box prop     = %4.2e seconds\n", dtime02 - dtime01);
	VRB.Result(cname, fname, "box contract = %4.2e seconds\n", dtime1 - dtime02);
	VRB.Result(cname, fname, "kl3 prop     = %4.2e seconds\n", dtime2 - dtime1);
	VRB.Result(cname, fname, "kl3          = %4.2e seconds\n", dtime3 - dtime2);
	VRB.Result(cname, fname, "k2pipi prop  = %4.2e seconds\n", dtime4 - dtime3);
	VRB.Result(cname, fname, "k2pipi       = %4.2e seconds\n", dtime5 - dtime4);
	VRB.Result(cname, fname, "total        = %4.2e seconds\n", dtime5 - dtime0);

	//////////////////////////////////////////////////////////////////////
	// store propagators
	// lprop_e.store_all("lprop_raw_", lqpropw_arg.cg.mass, traj);

	// Float dtime6 = dclock();

	// VRB.Result(cname, fname, "store prop   = %17.10e seconds\n", dtime6 - dtime5);
}

// Shift the locations of where exact propagators are calculated. This
// is done by shifting the following arrays,
//
// l_ut_loc, l_tw_loc
// s_ut_loc, s_tw_loc
void do_shift(int traj)
{
	const char* fname = "do_shift()";
	VRB.Result(cname, fname, "Shifting locations at which we will calculate exact propagators...\n");

	const int t_size = GJP.TnodeSites() * GJP.Tnodes();

	// We shift the exact solutions by a random amount uniformly
	// distributed between [0,T).
	int shift = drand48() * t_size;
	shift = shift - (shift%4);
  // int shift = 51;

	// Make sure we have the same number on all nodes.
	QMP_broadcast(&shift, sizeof(int));
	assert(shift >= 0 && shift < t_size);

	static int shift_acc = 0;

	shift_acc = (shift_acc + shift) % t_size;
	printf("traj = %d, Shift on the exact propagators = %d\n", traj, shift_acc);
	// VRB.Result(cname, fname, "traj = %d, Shift on the exact propagators = %d\n", traj, shift_acc);

	for(unsigned i = 0; i < l_ut_loc.v.v_len; ++i) {
		l_ut_loc.v.v_val[i] = (l_ut_loc.v.v_val[i] + shift) % t_size;
	}
	for(unsigned i = 0; i < l_tw_loc.v.v_len; ++i) {
		l_tw_loc.v.v_val[i] = (l_tw_loc.v.v_val[i] + shift) % t_size;
	}
	for(unsigned i = 0; i < s_ut_loc.v.v_len; ++i) {
		s_ut_loc.v.v_val[i] = (s_ut_loc.v.v_val[i] + shift) % t_size;
	}
	for(unsigned i = 0; i < s_tw_loc.v.v_len; ++i) {
		s_tw_loc.v.v_val[i] = (s_tw_loc.v.v_val[i] + shift) % t_size;
	}
}

int main(int argc,char *argv[])
{
	VRB.Level(VERBOSE_FUNC_LEVEL);;

	const char *fname = "main()";

	// Seed the random number generator this is used for shifting the source 
	// times of the exact propagators.
	srand48(time(NULL));
	// srand48(123456); //FIXME: fixed seed being used for testing
	// VRB.Result(cname, fname, "WARNING!!!! Using fixed random seed!!!!! FIX ME!!!!\n");

	//Test
	//More testing

	setup(argc, argv);


//	if(UniqueID() == 0) printf("after setup !!!\n");
//	int node_rank;
//	MPI_Comm_rank(QMP_COMM_WORLD, &node_rank);
//	printf("\tUniqueID(): %03d vs QMP_COMM_WORLD: %03d\n", UniqueID(), node_rank);

	VRB.Result(cname, fname, "Starting main trajectory loop\n");

	int traj = meas_arg.TrajStart;
	int ntraj = (meas_arg.TrajLessThanLimit - traj)/meas_arg.TrajIncrement + 1;

  for(int i = 0; i < argc; i++){
    if( std::strcmp(argv[i], "configuration-number") == 0 ){
      if(i+1 > argc){
        printf("wrong configuration-number argument(%d+1>%d).\n", i, argc);
        std::exit(1);
      }
      std::sscanf(argv[i+1], "%d", &traj);
      ntraj = 1;
      VRB.Result(cname, fname, "Doing measurement on ONE configuration(=%d)\n", traj); 
      i += 2;
      break;
    }
  }

	for(int conf = 0; conf < ntraj; ++conf) {
		VRB.Result(cname, fname, "Starting to work on trajectory %d\n", traj);

		GnoneFbfm lat;

		lat.SetBfmArg(lqpropw_arg.cg.mass);
	
//		int number_of_eigenvectors = 2000;
//		multi1d<Fermion_t[2]> _eigenvectors(number_of_eigenvectors);
//		multi1d<float> _eigenvalues(number_of_eigenvectors);

//		char eigenvector_stem[] = "/bgusr/data10/chulwoo/qcddata/DWF/2+1f/24nt64/IWASAKI+DSDR/b1.633/ls24/M1.8/ms0.0850/ml0.00107/evecs";
//		char eigenvector_stem[] = "/bgusr/home/jtu/24ID-evec/24D/job-0";
//		char eigenvector_directory[1024];
//		sprintf(eigenvector_directory, "%s/%d/lanczos.output", eigenvector_stem, traj);
//		sprintf(eigenvector_directory, "%s%d/lanczos.output", eigenvector_stem, traj);

//		char gf_matrices_stem[] = "/home/jiquntu/CSD/config/24x64x24ID/gf_matrices_COULOMB";
//		char gf_matrices_directory[1024];
//		sprintf(gf_matrices_directory, "%s.%d", gf_matrices_stem, traj);

//		load_eigenvectors_luchang(lat, eigenvector_directory, _eigenvectors, _eigenvalues);
		// load_eigenvectors(lat, eigenvector_directory, eigenvectors, eigenvalues);

//    	Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].eigenvectors = &_eigenvectors;
//    	Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].eigenvalues = &_eigenvalues;

//		lat.set_deflation(&eigenvectors, &eigenvalues, 0); // !!! The last 0 is the number of low modes to be subtracted from solutions, NOT the number of eigenvectors.

		// shift the exact propagators
		do_shift(traj);
		load_checkpoint(lat, traj);
		
//		run_ritz(lat);
//		test_local_deflation(lat);

//		multi1d<float> matrix_elements(number_of_eigenvectors);
//		test_eigenvectors(lat, eigenvectors, eigenvalues, matrix_elements);
		// low_mode_submatrix(lat, eigenvectors, eigenvalues);
		
		// NOTE: there are 4 twists but 3 momenta. This is just
		// because of how code is written. The t twist is normally
		// zero.
		assert(l_twist_arg.Floats.Floats_len == 4);
		assert(s_twist_arg.Floats.Floats_len == 4);
		assert(d_mom_kpp.v.v_len == 3);

		const double *ltw = l_twist_arg.Floats.Floats_val;
		const double *stw = s_twist_arg.Floats.Floats_val;
		const int *dmom = d_mom_kpp.v.v_val;

		VRB.Result(cname, fname,
				"l quark twist (kl3) = %17.10e %17.10e %17.10e %17.10e\n",
				ltw[0], ltw[1], ltw[2], ltw[3]);
		VRB.Result(cname, fname,
				"s quark twist (kl3) = %17.10e %17.10e %17.10e %17.10e\n",
				stw[0], stw[1], stw[2], stw[3]);
		VRB.Result(cname, fname,
				"d quark mom  (k2pp) = %d %d %d\n",
				dmom[0], dmom[1], dmom[2]);

		// run_all(lat, stw, ltw, dmom, traj, gf_matrices_directory);
		run_all(lat, stw, ltw, dmom, traj);

		traj += meas_arg.TrajIncrement;
	
		// Release the memory allocated in loading_eigenvectors_luchang().
//		for(int i = 0; i < number_of_eigenvectors; i++){
//			delete [] _eigenvectors[i][1];
//		}
	}

	VRB.Result(cname, fname, "Program ended normally.\n");
	End();
}

void load_checkpoint(Lattice& lat, int traj)
{
	const char *fname = "load_checkpoint()";

	char lat_file[256];
	// GnoneFnone lat;

	sprintf(lat_file, "%s.%d", meas_arg.GaugeStem, traj);
	QioArg rd_arg(lat_file, 0.001);
	rd_arg.ConcurIONumber = meas_arg.IOconcurrency;
	ReadLatticeParallel rl;
	rl.read(lat,rd_arg);
	if(!rl.good()) ERR.General(cname,fname,"Failed read lattice %s\n",lat_file);
}

void setup(int argc, char *argv[])
{
	const char *fname = "setup()";

	Start(&argc, &argv);

//	if(UniqueID() == 0) printf("within setup !!!\n");
//	int node_rank;
//	MPI_Comm_rank(QMP_COMM_WORLD, &node_rank);
//	printf("\tUniqueID(): %03d: %02dx%02dx%02dx%02d vs QMP_COMM_WORLD: %03d\n", UniqueID(), CoorX(), CoorY(), CoorZ(), CoorT(), node_rank);

	if(argc < 2) {
		ERR.General(cname, fname, "Must provide VML directory.\n");
	}

	if(chdir(argv[1]) != 0) {
		ERR.General(cname, fname, "Changing directory to %s failed.\n", argv[1]);
	}

	encode_vml_all();
	decode_vml_all();

	if(chdir(meas_arg.WorkDirectory) != 0) {
		ERR.General(cname, fname, "Changing directory to %s failed.\n", meas_arg.WorkDirectory);
	}
	VRB.Result(cname, fname, "Reading VML files successfully.\n");

	if(UniqueID() == 0){
		mkdir("../results", 0755);
		mkdir("../resultsA", 0755);
		mkdir("../resultsEA", 0755);
		mkdir("../resultsEP", 0755);
		mkdir("../resultsEPA", 0755);
		mkdir("../resultsP", 0755);
		mkdir("../resultsPA", 0755);
	}

	GJP.Initialize(do_arg);
	LRG.Initialize();

	GJP.InitializeExt(doext_arg);
	
	// qlat::begin(&argc, &argv); // qlat init

 	qlat::begin(QMP_COMM_WORLD, qlat::Coordinate(SizeX(), SizeY(), SizeZ(), SizeT()));
	
	init_bfm(&argc, &argv);
}

void use_omegas(bfmarg &ba, std::vector<std::complex<double> > &omegas)
{
    const char* fname = "use_omegas";

    int Ls = ba.Ls;
    assert(Ls == omegas.size());

    ba.solver = HmCayleyComplex;
    for (int s = 0; s < Ls; s++) {
	std::complex<double> b_s = 0.5 * (1.0 / omegas[s] + 1.0);
	std::complex<double> c_s = 0.5 * (1.0 / omegas[s] - 1.0);

	VRB.Result(cname, fname, "b[s=%d] = %0.10e + i %0.10e, c[s=%d] = %0.10e + i %0.10e\n", s, b_s.real(), b_s.imag(), s, c_s.real(), c_s.imag());

	ba.bs_[s] = b_s.real();
	ba.bsi_[s] = b_s.imag();
	ba.cs_[s] = c_s.real();
	ba.csi_[s] = c_s.imag();
    }
}

void init_bfm(int *argc, char **argv[])
{
	const char* fname = "init_bfm()";

	QDP::QDP_initialize(argc, argv);

	// Chroma::initialize(argc, argv);
	multi1d<int> nrow(Nd);

	for(int i = 0; i< Nd; ++i){
		nrow[i] = GJP.Sites(i);
	}

	Layout::setLattSize(nrow);
	Layout::create();
	
	bfmarg::Threads(64);
	bfmarg::Reproduce(0);
	bfmarg::ReproduceChecksum(0);
	bfmarg::ReproduceMasterCheck(0);
	bfmarg::Verbose(0);
	Fbfm::use_mixed_solver = true;

/*
// Directly using zMobius

	const Float M5 = GJP.DwfHeight();
	const int unitary_Ls = GJP.Sites(4); // should be 12
	const Float mobius_scale = 1.0;

	std::vector<std::complex<double> > omegas(12);
	omegas[0] = 1.0903256131299373;
	omegas[1] = 0.9570283702230611;
	omegas[2] = 0.7048886040934104;
	omegas[3] = 0.48979921782791747;
	omegas[4] = 0.328608311201356;
	omegas[5] = 0.21664245377015995;
	omegas[6] = 0.14121112711957107;
	omegas[7] = 0.0907785101745156;
	omegas[8] = std::complex<double>( 0.05608303440064219, -0.007537158177840385 );
	omegas[9] = std::complex<double>( 0.05608303440064219, 0.007537158177840385 );
	omegas[10] =std::complex<double>( 0.0365221637144842, -0.03343945161367745 );
	omegas[11] =std::complex<double>( 0.0365221637144842, 0.03343945161367745 ); 
   
   	Fbfm::arg_map[sqpropw_arg.cg.mass].ScaledShamirCayleyTanh(sqpropw_arg.cg.mass, M5, unitary_Ls, mobius_scale);
    use_omegas(Fbfm::arg_map[sqpropw_arg.cg.mass], omegas);
	Fbfm::arg_map[sqpropw_arg.cg.mass].CGdiagonalMee = 2;
	
	Fbfm::arg_map[lqpropw_arg.cg.mass].ScaledShamirCayleyTanh(lqpropw_arg.cg.mass, M5, unitary_Ls, mobius_scale);
    use_omegas(Fbfm::arg_map[lqpropw_arg.cg.mass], omegas);
	Fbfm::arg_map[lqpropw_arg.cg.mass].CGdiagonalMee = 2;

// End zMobius
*/


// MADWF by Jackson
//
// The DWF setting here is suppose to be unitary, i.e. the setting that was used to generate the ensemble.

	const Float M5 = GJP.DwfHeight();
	const int unitary_Ls = GJP.Sites(4);
	const Float mobius_scale = 2.0;


	Fbfm::arg_map[sqpropw_arg.cg.mass].ScaledShamirCayleyTanh(sqpropw_arg.cg.mass, M5, unitary_Ls, mobius_scale);
	//Fbfm::arg_map[sqpropw_arg.cg.mass].CGdiagonalMee = 2;
	Fbfm::arg_map[lqpropw_arg.cg.mass].ScaledShamirCayleyTanh(lqpropw_arg.cg.mass, M5, unitary_Ls, mobius_scale);
	//Fbfm::arg_map[lqpropw_arg.cg.mass].CGdiagonalMee = 2;
// MADWF Setting

/*

// No MADWF for heavy quark. 

// Now the DWF setting we are aimming to have, supposely with smaller Ls.
//	const int MADWF_Ls = 14;
//	std::vector<std::complex<double> > madwf_omegas(14);
//    madwf_omegas[0] = 1.395566319;
//    madwf_omegas[1] = 0.8734555838;
//    madwf_omegas[2] = 0.5976868047;
//    madwf_omegas[3] = 0.3575610385;
//    madwf_omegas[4] = 0.1765657746;
//    madwf_omegas[5] = std::complex<double>(0.09298492219, +0.02404746984);
//    madwf_omegas[6] = std::complex<double>(0.06049178877, +0.06547113295);
//    madwf_omegas[7] = std::complex<double>(0.06049178877, -0.06547113295);
//    madwf_omegas[8] = std::complex<double>(0.09298492219, -0.02404746984);
//    madwf_omegas[9] = 0.119596635;
//    madwf_omegas[10] = 0.2566858057;
//    madwf_omegas[11] = 0.4745730595;
//    madwf_omegas[12] = 0.6986654036;
//    madwf_omegas[13] = 1.217162021;
 
	const int MADWF_Ls = 12;
	std::vector<std::complex<double> > madwf_omegas(12);
	madwf_omegas[0] = 1.0903256131299373;
	madwf_omegas[1] = 0.9570283702230611;
	madwf_omegas[2] = 0.7048886040934104;
	madwf_omegas[3] = 0.48979921782791747;
	madwf_omegas[4] = 0.328608311201356;
	madwf_omegas[5] = 0.21664245377015995;
	madwf_omegas[6] = 0.14121112711957107;
	madwf_omegas[7] = 0.0907785101745156;
	madwf_omegas[8] = std::complex<double>( 0.05608303440064219, -0.007537158177840385 );
	madwf_omegas[9] = std::complex<double>( 0.05608303440064219, 0.007537158177840385 );
	madwf_omegas[10] =std::complex<double>( 0.0365221637144842, -0.03343945161367745 );
	madwf_omegas[11] =std::complex<double>( 0.0365221637144842, 0.03343945161367745 ); 
*/
/* 
	Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].cheap_approx.ScaledShamirCayleyTanh(sqpropw_arg.cg.mass, M5, MADWF_Ls, mobius_scale);
    use_omegas(Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].cheap_approx, madwf_omegas);
    Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].cheap_approx.CGdiagonalMee = 2;
    Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].num_dc_steps = 1;
    Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].cheap_solve_stop_rsd = 1e-8;
    Fbfm::madwf_arg_map[sqpropw_arg.cg.mass].exact_pv_stop_rsd = 1e-8;
*/
  
/*
	Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].cheap_approx.ScaledShamirCayleyTanh(lqpropw_arg.cg.mass, M5, MADWF_Ls, mobius_scale);
    use_omegas(Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].cheap_approx, madwf_omegas);
    Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].cheap_approx.CGdiagonalMee = 2;
    Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].num_dc_steps = 1;
    Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].cheap_solve_stop_rsd = 1e-8;
    Fbfm::madwf_arg_map[lqpropw_arg.cg.mass].exact_pv_stop_rsd = 1e-8;
*/
	VRB.Result(cname, fname, "init_bfm finished successfully\n");
}

