#pragma once

#include <qlat/qlat.h>

QLAT_START_NAMESPACE

typedef std::complex<float> ComplexF;

template<class T>
void caxpy_single(std::complex<T>* res, const std::complex<T>& ca, const std::complex<T>* x, const std::complex<T>* y, const int c_size)
{
  for (int i=0;i<c_size;i++) {
    res[i] = ca*x[i] + y[i];
  }
}

inline void read_floats(Vector<float> out, const Vector<uint8_t> fp_data)
{
  qassert(out.data_size() == fp_data.size());
  memcpy(out.data(), fp_data.data(), fp_data.size());
  to_from_little_endian_32(out);
}

inline int fp_map(const float in, const float min, const float max, const int N)
{
  // Idea:
  //
  // min=-6
  // max=6
  //
  // N=1
  // [-6,0] -> 0, [0,6] -> 1;  reconstruct 0 -> -3, 1-> 3
  //
  // N=2
  // [-6,-2] -> 0, [-2,2] -> 1, [2,6] -> 2;  reconstruct 0 -> -4, 1->0, 2->4
  int ret =  (int) ( (float)(N+1) * ( (in - min) / (max - min) ) );
  if (ret == N+1) {
    ret = N;
  }
  return ret;
}

inline float fp_unmap(const int val, const float min, const float max, const int N)
{
  return min + (float)(val + 0.5) * (max - min)  / (float)( N + 1 );
}

inline float unmap_fp16_exp(unsigned short e)
{
  const double base = 1.4142135623730950488;
  const float de = (float)((int)e - USHRT_MAX/ 2);
  return pow( base, de );
}

inline long fp_16_size(const long f_size, const long nsc)
{
  return (f_size + f_size / nsc) * sizeof(uint16_t);
}

inline void read_floats_fp16(float* out, const uint8_t* ptr, const int64_t n, const int nsc)
{
  const int64_t nsites = n / nsc;
  qassert(n % nsc == 0);
  const unsigned short* in = (const unsigned short*)ptr;
  // do for each site
  for (int64_t site = 0;site<nsites;site++) {
    float* ev = &out[site*nsc];
    const unsigned short* bptr = &in[site*(nsc + 1)];
    const unsigned short exp = bptr[0];
    const float max = unmap_fp16_exp(exp);
    const float min = -max;
    for (int i=0;i<nsc;i++) {
      ev[i] = fp_unmap(bptr[1+i], min, max, USHRT_MAX);
    }
  }
}

inline void read_floats_fp16(Vector<float> out, const Vector<uint8_t> fp_data, const int nsc)
{
  const long size = fp_16_size(out.size(), nsc);
  qassert(fp_data.size() == size);
  std::vector<uint16_t> buffer(size / sizeof(uint16_t));
  memcpy(buffer.data(), fp_data.data(), fp_data.size());
  to_from_little_endian_16(get_data(buffer));
  read_floats_fp16(out.data(), (const uint8_t*)buffer.data(), out.size(), nsc);
}

struct CompressedEigenSystemInfo
{
  // int s[5]; // local vol size
  // int b[5]; // block size
  // // derived
  // int nb[5];
  // int blocks;
  // //
  // int index;
  //
  int nkeep_fp16; // nkeep - nkeep_single
  Coordinate total_node; // number of node in each direction (total)
  Coordinate total_block; // number of block in each direction (total)
  Coordinate node_block; // number of block in a node in each direction (node)
  //
  Coordinate total_site; // number of site in each direction (total)
  Coordinate node_site; // number of site in a node in each direction (node)
  Coordinate block_site; // number of site in each block in each direction (block)
  int ls;
  int neig;
  int nkeep; // number base
  int nkeep_single; // number base stored as single precision
  int FP16_COEF_EXP_SHARE_FLOATS;
  //
  std::vector<crc32_t> crcs;
};

struct CompressedEigenSystemDenseInfo
{
  Coordinate total_site;
  Coordinate node_site;
  Coordinate block_site;
  int ls;
  int neig;
  int nkeep;
  int nkeep_single;
  int FP16_COEF_EXP_SHARE_FLOATS;
};

inline CompressedEigenSystemInfo populate_eigen_system_info(const CompressedEigenSystemDenseInfo& cesdi, const std::vector<crc32_t>& crcs)
{
  CompressedEigenSystemInfo cesi;
  cesi.crcs = crcs;
  //
  cesi.total_site = cesdi.total_site;
  cesi.node_site = cesdi.node_site;
  cesi.block_site = cesdi.block_site;
  cesi.ls = cesdi.ls;
  cesi.neig = cesdi.neig;
  cesi.nkeep = cesdi.nkeep;
  cesi.nkeep_single = cesdi.nkeep_single;
  cesi.FP16_COEF_EXP_SHARE_FLOATS = cesdi.FP16_COEF_EXP_SHARE_FLOATS;
  //
  qassert(cesi.nkeep >= cesi.nkeep_single);
  cesi.nkeep_fp16 = cesi.nkeep - cesi.nkeep_single;
  cesi.total_node = cesi.total_site / cesi.node_site;
  cesi.total_block = cesi.total_site / cesi.block_site;
  cesi.node_block = cesi.node_site / cesi.block_site;
  qassert(cesi.total_site == cesi.total_node * cesi.node_site);
  qassert(cesi.total_site == cesi.total_block * cesi.block_site);
  qassert(cesi.node_site == cesi.node_block * cesi.block_site);
  qassert((int)cesi.crcs.size() == product(cesi.total_node));
  return cesi;
}

inline CompressedEigenSystemInfo read_compressed_eigen_system_info(const std::string& root)
{
  TIMER_VERBOSE("read_compressed_eigen_system_info");
  CompressedEigenSystemDenseInfo cesdi;
  std::vector<crc32_t> crcs;
  if (get_id_node() == 0) {
    std::vector<std::string> lines = qgetlines(root + "/metadata.txt");
    for (int i = 0; i < 4; ++i) {
      reads(cesdi.node_site[i], info_get_prop(lines, ssprintf("s[%d] = ", i)));
      reads(cesdi.block_site[i], info_get_prop(lines, ssprintf("b[%d] = ", i)));
      int nb;
      reads(nb, info_get_prop(lines, ssprintf("nb[%d] = ", i)));
      qassert(cesdi.node_site[i] == nb * cesdi.block_site[i]);
    }
    reads(cesdi.ls, info_get_prop(lines, ssprintf("s[4] = ")));
    int b5, nb5;
    reads(b5, info_get_prop(lines, ssprintf("b[4] = ")));
    reads(nb5, info_get_prop(lines, ssprintf("nb[4] = ")));
    qassert(b5 == cesdi.ls);
    qassert(nb5 == 1);
    reads(cesdi.neig, info_get_prop(lines, "neig = "));
    reads(cesdi.nkeep, info_get_prop(lines, "nkeep = "));
    reads(cesdi.nkeep_single, info_get_prop(lines, "nkeep_single = "));
    int blocks;
    reads(blocks, info_get_prop(lines, "blocks = "));
    qassert(blocks == product(cesdi.node_site / cesdi.block_site));
    reads(cesdi.FP16_COEF_EXP_SHARE_FLOATS, info_get_prop(lines, "FP16_COEF_EXP_SHARE_FLOATS = "));
    for (long idx = 0; true; ++idx) {
      const std::string value = info_get_prop(lines, ssprintf("crc32[%d] = ", idx));
      if (value == "") {
        break;
      }
      crcs.push_back(read_crc32(value));
    }
    const long global_volume = crcs.size() * product(cesdi.node_site);
    if (64*64*64*128 == global_volume) {
      cesdi.total_site = Coordinate(64, 64, 64, 128);
    } else if (48*48*48*96 == global_volume) {
      cesdi.total_site = Coordinate(48, 48, 48, 96);
    } else if (48*48*48*64 == global_volume) {
      cesdi.total_site = Coordinate(48, 48, 48, 64);
    } else if (24*24*24*64 == global_volume) {
      cesdi.total_site = Coordinate(24, 24, 24, 64);
    } else if (32*32*32*64 == global_volume) {
      cesdi.total_site = Coordinate(32, 32, 32, 64);
    } else {
      cesdi.total_site = Coordinate();
    }
    for (int i = 0; i < 4; ++i) {
      const std::string str_gs = info_get_prop(lines, ssprintf("gs[%d] = ", i));
      if (str_gs != "") {
        reads(cesdi.total_site[i], str_gs);
      }
    }
    const std::string str_gs5 = info_get_prop(lines, ssprintf("gs[4] = "));
    if (str_gs5 != "") {
      int gs5;
      reads(gs5, str_gs5);
      qassert(cesdi.ls == gs5);
    }
    qassert(product(cesdi.total_site) == global_volume);
  }
  bcast(Vector<CompressedEigenSystemDenseInfo>(&cesdi, 1));
  crcs.resize(product(cesdi.total_site / cesdi.node_site));
  bcast(get_data(crcs));
  return populate_eigen_system_info(cesdi, crcs);
}

struct HalfVector : Field<ComplexF>
{
  static const int c_size = 12; // number of complex number per wilson vector
  int ls;
  // geo.multiplicity = ls * c_size;
  virtual const std::string& cname()
  {
    static const std::string s = "HalfVector";
    return s;
  }
};

inline void init_half_vector(HalfVector& hv, const Geometry& geo, const int ls)
  // eo = odd
{
  TIMER("init_half_vector");
  hv.ls = ls;
  const Geometry geo_odd = geo_eo(geo_reform(geo, ls * HalfVector::c_size, 0), 1);
  hv.init();
  hv.init(geo_odd);
}

struct CompressedEigenSystemData : Field<uint8_t>
{
  CompressedEigenSystemInfo cesi;
  long block_vol_eo;
  int ls;
  long basis_c_size;
  long coef_c_size;
  long basis_size_single;
  long basis_size_fp16;
  long coef_size_single;
  long coef_size_fp16;
  long coef_size;
  long bases_size_single;
  long bases_size_fp16;
  long bases_offset_single;
  long bases_offset_fp16;
  long coefs_offset;
  long end_offset;
  //
  virtual const std::string& cname()
  {
    static const std::string s = "CompressedEigenSystemData";
    return s;
  }
  //
};

inline Geometry get_geo_from_cesi(const CompressedEigenSystemInfo& cesi, const int id_node,
    const Coordinate& new_size_node = Coordinate())
{
  GeometryNode geon;
  Coordinate node_site;
  if (new_size_node == Coordinate()) {
    geon.size_node = cesi.total_site / cesi.node_site;
  } else {
    geon.size_node = new_size_node;
    node_site = cesi.total_site / new_size_node;
    qassert(cesi.total_site == new_size_node * node_site);
  }
  geon.id_node = id_node;
  geon.num_node = product(geon.size_node);
  geon.coor_node = coordinate_from_index(geon.id_node, geon.size_node);
  Geometry geo_full;
  geo_full.init(geon, 1, node_site);
  return geo_full;
};

inline Geometry block_geometry(const Geometry& geo_full, const Coordinate& block_site)
  // new multiplicity = 1
{
  qassert(geo_full.node_site % block_site == Coordinate());
  const Coordinate node_block = geo_full.node_site / block_site;
  Geometry geo;
  geo.init(geo_full.geon, 1, node_block);
  return geo;
}

inline void init_compressed_eigen_system_data(CompressedEigenSystemData& cesd,
    const CompressedEigenSystemInfo& cesi, const int id_node, const Coordinate& new_size_node = Coordinate())
{
  TIMER("init_compressed_eigen_system_data");
  cesd.cesi = cesi;
  cesd.block_vol_eo = product(cesi.block_site)/2;
  cesd.ls = cesi.ls;
  cesd.basis_c_size = cesd.block_vol_eo * cesd.ls * HalfVector::c_size;
  cesd.coef_c_size = cesi.nkeep;
  cesd.basis_size_single = cesd.basis_c_size * sizeof(ComplexF);
  cesd.basis_size_fp16 = fp_16_size(cesd.basis_c_size * 2, 24);
  cesd.coef_size_single = cesi.nkeep_single * sizeof(ComplexF);
  cesd.coef_size_fp16 = fp_16_size(cesi.nkeep_fp16 * 2, cesi.FP16_COEF_EXP_SHARE_FLOATS);
  cesd.coef_size = cesd.coef_size_single + cesd.coef_size_fp16;
  cesd.bases_size_single = cesi.nkeep_single * cesd.basis_size_single;
  cesd.bases_size_fp16 = cesi.nkeep_fp16 * cesd.basis_size_fp16;
  cesd.bases_offset_single = 0;
  cesd.bases_offset_fp16 = cesd.bases_offset_single + cesd.bases_size_single;
  cesd.coefs_offset = cesd.bases_offset_fp16 + cesd.bases_size_fp16;
  cesd.end_offset = cesd.coefs_offset + cesi.neig * cesd.coef_size;
  const Geometry geo_full = get_geo_from_cesi(cesi, id_node, new_size_node);
  const Geometry geo = geo_remult(block_geometry(geo_full, cesi.block_site), cesd.end_offset);
  cesd.init(geo);
}

struct CompressedEigenSystemBases : Field<ComplexF>
{
  int n_basis;
  long block_vol_eo; // even odd precondition
                     // (block_vol_eo is half of the number of site within the block n_t * n_z * n_y * n_x/2)
                     // n_t slowest varying and n_x fastest varying
  int ls; // ls varying faster than n_x
  int c_size_vec; // c_size_vec = block_vol_eo * ls * HalfVector::c_size;
  // geo.multiplicity = n_basis * c_size_vec;
  Geometry geo_full;
  Coordinate block_site;
  //
  virtual const std::string& cname()
  {
    static const std::string s = "CompressedEigenSystemBases";
    return s;
  }
};

struct CompressedEigenSystemCoefs : Field<ComplexF>
{
  int n_vec;
  int n_basis;
  int ls;
  int c_size_vec; // c_size_vec = n_basis;
  // geo.multiplicity = n_vec * c_size_vec;
  Geometry geo_full;
  Coordinate block_site;
  //
  virtual const std::string& cname()
  {
    static const std::string s = "CompressedEigenSystemCoefs";
    return s;
  }
};

inline void init_compressed_eigen_system_bases(
    CompressedEigenSystemBases& cesb,
    const int n_basis,
    const Geometry& geo_full,
    const Coordinate& block_site,
    const int ls)
{
  TIMER("init_compressed_eigen_system_bases");
  cesb.n_basis = n_basis;
  cesb.block_vol_eo = product(block_site) / 2;
  cesb.ls = ls;
  cesb.c_size_vec = cesb.block_vol_eo * cesb.ls * HalfVector::c_size;
  const Geometry geo = geo_remult(block_geometry(geo_full, block_site), n_basis * cesb.c_size_vec);
  cesb.geo_full = geo_full;
  cesb.block_site = block_site;
  cesb.init();
  cesb.init(geo);
}

inline void init_compressed_eigen_system_bases(
    CompressedEigenSystemBases& cesb, const CompressedEigenSystemInfo& cesi, const int id_node,
    const Coordinate& new_size_node = Coordinate())
{
  const Geometry geo_full = get_geo_from_cesi(cesi, id_node, new_size_node);
  init_compressed_eigen_system_bases(cesb, cesi.nkeep, geo_full, cesi.block_site, cesi.ls);
}

inline void init_compressed_eigen_system_coefs(
    CompressedEigenSystemCoefs& cesc,
    const int n_vec,
    const int n_basis,
    const Geometry& geo_full,
    const Coordinate& block_site,
    const int ls)
{
  TIMER("init_compressed_eigen_system_coefs");
  cesc.n_vec = n_vec;
  cesc.n_basis = n_basis;
  cesc.c_size_vec = n_basis;
  const Geometry geo = geo_remult(block_geometry(geo_full, block_site), n_vec * cesc.c_size_vec);
  cesc.geo_full = geo_full;
  cesc.block_site = block_site;
  cesc.ls = ls;
  cesc.init();
  cesc.init(geo);
}

inline void init_compressed_eigen_system_coefs(
    CompressedEigenSystemCoefs& cesc, const CompressedEigenSystemInfo& cesi, const int id_node,
    const Coordinate& new_size_node = Coordinate())
{
  const Geometry geo_full = get_geo_from_cesi(cesi, id_node, new_size_node);
  init_compressed_eigen_system_coefs(cesc, cesi.neig, cesi.nkeep, geo_full, cesi.block_site, cesi.ls);
}

struct VFile
{
  std::string fn;
  std::string mode;
  long pos;
  long size;
  std::multimap<long, Vector<uint8_t> > read_entries;
};

inline VFile vopen(const std::string& fn, const std::string& mode)
{
  VFile fp;
  fp.fn = fn;
  fp.mode = mode;
  fp.pos = 0;
  fp.size = -1;
  fp.read_entries.clear();
  return fp;
}

inline void vclose(VFile& fp)
{
  if (not fp.read_entries.empty()) {
    TIMER_VERBOSE_FLOPS("vclose");
    FILE* fpr = qopen(fp.fn, fp.mode);
    for (std::multimap<long, Vector<uint8_t> >::const_iterator it = fp.read_entries.cbegin(); it != fp.read_entries.cend(); ++it) {
      fseek(fpr, it->first, SEEK_SET);
      qread_data(it->second, fpr);
      timer.flops += it->second.size();
    }
    qclose(fpr);
  }
  fp.pos = 0;
  fp.read_entries.clear();
}

inline void set_vfile_size(VFile& fp)
{
  if (fp.size < 0) {
    FILE* fpr = qopen(fp.fn, fp.mode);
    fseek(fpr, 0, SEEK_END);
    fp.size = ftell(fpr);
    qclose(fpr);
  }
}

template <class M>
long vread_data(const Vector<M>& v, VFile& fp)
{
  set_vfile_size(fp);
  Vector<uint8_t> dv((uint8_t*)v.data(), v.data_size());
  fp.read_entries.insert(std::pair<long, Vector<uint8_t> >(fp.pos, dv));
  fp.pos += dv.size();
  qassert(0 <= fp.pos && fp.pos <= fp.size);
  return dv.size();
}

inline int vseek(VFile& fp, const long offset, const int whence)
{
  set_vfile_size(fp);
  if (whence == SEEK_SET) {
    fp.pos = offset;
  } else if (whence == SEEK_CUR) {
    fp.pos += offset;
  } else if (whence == SEEK_END) {
    fp.pos = fp.size + offset;
  } else {
    qassert(false);
  }
  qassert(0 <= fp.pos && fp.pos <= fp.size);
  return 0;
}

inline long vtell(const VFile& fp)
{
  return fp.pos;
}

inline void load_block_data(
    CompressedEigenSystemData& cesd, const Coordinate& xl,
    const CompressedEigenSystemInfo& cesi,
    VFile& fp, const Coordinate& xl_file)
{
  TIMER("load_block_data");
  Vector<uint8_t> data = cesd.get_elems(xl);
  const int block_idx = index_from_coordinate(xl_file, cesi.node_block);
  const int block_size = product(cesi.node_block);
  vseek(fp, 0, SEEK_END);
  const long file_size = vtell(fp);
  qassert(file_size == block_size * cesd.end_offset);
  const long bases_offset_single = block_size * cesd.bases_offset_single;
  const long bases_offset_fp16 = block_size * cesd.bases_offset_fp16;
  const long coefs_offset = block_size * cesd.coefs_offset;
  {
    vseek(fp, bases_offset_single + block_idx * cesd.bases_size_single, SEEK_SET);
    Vector<uint8_t> chunk(&data[cesd.bases_offset_single], cesd.bases_size_single);
    vread_data(chunk, fp);
  }
  {
    vseek(fp, bases_offset_fp16 + block_idx * cesd.bases_size_fp16, SEEK_SET);
    Vector<uint8_t> chunk(&data[cesd.bases_offset_fp16], cesd.bases_size_fp16);
    vread_data(chunk, fp);
  }
  for (int i = 0; i < cesi.neig; ++i) {
    vseek(fp, coefs_offset + i * block_size * cesd.coef_size + block_idx * cesd.coef_size, SEEK_SET);
    Vector<uint8_t> chunk(&data[cesd.coefs_offset + i * cesd.coef_size], cesd.coef_size);
    vread_data(chunk, fp);
  }
}

inline crc32_t load_block_data_crc(
    const CompressedEigenSystemData& cesd, const Coordinate& xl,
    const CompressedEigenSystemInfo& cesi, const Coordinate& xl_file)
{
  TIMER("load_block_data_crc");
  crc32_t crc = 0;
  const Vector<uint8_t> data = cesd.get_elems_const(xl);
  const int block_idx = index_from_coordinate(xl_file, cesi.node_block);
  const int block_size = product(cesi.node_block);
  const long file_size = block_size * cesd.end_offset;
  const long bases_offset_single = block_size * cesd.bases_offset_single;
  const long bases_offset_fp16 = block_size * cesd.bases_offset_fp16;
  const long coefs_offset = block_size * cesd.coefs_offset;
  {
    const Vector<uint8_t> chunk(&data[cesd.bases_offset_single], cesd.bases_size_single);
    crc ^= crc32_combine(crc32(chunk), 0,
        file_size - (bases_offset_single + (block_idx + 1) * cesd.bases_size_single));
  }
  {
    const Vector<uint8_t> chunk(&data[cesd.bases_offset_fp16], cesd.bases_size_fp16);
    crc ^= crc32_combine(crc32(chunk), 0,
        file_size - (bases_offset_fp16 + (block_idx + 1) * cesd.bases_size_fp16));
  }
  for (int i = 0; i < cesi.neig; ++i) {
    const Vector<uint8_t> chunk(&data[cesd.coefs_offset + i * cesd.coef_size], cesd.coef_size);
    crc ^= crc32_combine(crc32(chunk), 0,
        file_size - (coefs_offset + i * block_size * cesd.coef_size + (block_idx + 1) * cesd.coef_size));
  }
  return crc;
}

inline void load_block(
    CompressedEigenSystemBases& cesb, CompressedEigenSystemCoefs& cesc,
    const CompressedEigenSystemData& cesd, const Coordinate& xl,
    const CompressedEigenSystemInfo& cesi)
{
  TIMER_VERBOSE("load_block");
  Vector<ComplexF> bases = cesb.get_elems(xl);
  Vector<ComplexF> coefs = cesc.get_elems(xl);
  const Vector<uint8_t> data = cesd.get_elems_const(xl);
  {
    const long c_size = cesi.nkeep_single * cesb.c_size_vec;
    const long d_size = c_size * sizeof(ComplexF);
    read_floats(
        Vector<float>((float*)&bases[0], c_size * 2),
        Vector<uint8_t>(&data[cesd.bases_offset_single], d_size));
  }
  {
    const long c_size = cesi.nkeep_fp16 * cesb.c_size_vec;
    const long d_size = fp_16_size(c_size * 2, 24);
    read_floats_fp16(
        Vector<float>((float*)&bases[cesi.nkeep_single * cesb.c_size_vec], c_size * 2),
        Vector<uint8_t>(&data[cesd.bases_offset_fp16], d_size),
        24);
  }
#pragma omp parallel for
  for (int i = 0; i < cesb.n_basis; ++i) {
    qassert(cesb.c_size_vec == cesb.ls * cesb.block_vol_eo * HalfVector::c_size);
    Vector<ComplexF> basis(&bases[i * cesb.c_size_vec], cesb.c_size_vec);
    std::vector<ComplexF> buffer(cesb.c_size_vec);
    for (int s = 0; s < cesb.ls; ++s) {
      for (long index = 0; index < cesb.block_vol_eo; ++index) {
        memcpy(
            &buffer[(index * cesb.ls + s) * HalfVector::c_size],
            &basis[(s * cesb.block_vol_eo + index) * HalfVector::c_size],
            HalfVector::c_size * sizeof(ComplexF));
      }
    }
    assign(basis, get_data(buffer));
  }
  for (int i = 0; i < cesc.n_vec; ++i) {
    Vector<float> coef((float*)&coefs[i * cesc.c_size_vec], cesc.c_size_vec * 2);
    Vector<uint8_t> dc(&data[cesd.coefs_offset + i * cesd.coef_size], cesd.coef_size);
    read_floats(
        Vector<float>(&coef[0], cesi.nkeep_single * 2),
        Vector<uint8_t>(&dc[0], cesi.nkeep_single * sizeof(ComplexF)));
    read_floats_fp16(
        Vector<float>(&coef[cesi.nkeep_single * 2], cesi.nkeep_fp16 * 2),
        Vector<uint8_t>(&dc[cesi.nkeep_single * sizeof(ComplexF)],
          fp_16_size(cesi.nkeep_fp16 * 2, cesi.FP16_COEF_EXP_SHARE_FLOATS)),
        cesi.FP16_COEF_EXP_SHARE_FLOATS);
  }
}

inline std::vector<crc32_t> load_node(
    CompressedEigenSystemBases& cesb, CompressedEigenSystemCoefs& cesc,
    const CompressedEigenSystemInfo& cesi,
    const std::string& path)
{
  TIMER_VERBOSE("load_node");
  qassert(geo_remult(cesb.geo) == geo_remult(cesc.geo));
  std::vector<crc32_t> crcs(product(cesi.total_node), 0);
  const Geometry& geo = cesb.geo;
  CompressedEigenSystemData cesd;
  init_compressed_eigen_system_data(cesd, cesi, geo.geon.id_node, geo.geon.size_node);
  const int idx_size = product(cesi.total_node);
  std::vector<VFile> fps(idx_size);
  for (int idx = 0; idx < idx_size; ++idx) {
    const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
    fps[idx] = vopen(path + ssprintf("/%02d/%010d.compressed", dir_idx, idx), "r");
  }
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Coordinate xg = geo.coordinate_g_from_l(xl);
    const Coordinate xl_file = xg % cesi.node_block;
    const int idx = index_from_coordinate(xg / cesi.node_block, cesi.total_node);
    qassert(0 <= idx && idx < idx_size);
    displayln_info("load: fn='" + fps[idx].fn + "' ; index=" + show(index) + " ; xl=" + show(xl) + "/" + show(cesi.node_block));
    load_block_data(cesd, xl, cesi, fps[idx], xl_file);
  }
  for (int idx = 0; idx < idx_size; ++idx) {
    vclose(fps[idx]);
  }
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    const Coordinate xg = geo.coordinate_g_from_l(xl);
    const Coordinate xl_file = xg % cesi.node_block;
    const int idx = index_from_coordinate(xg / cesi.node_block, cesi.total_node);
    qassert(0 <= idx && idx < idx_size);
    crc32_t crc = load_block_data_crc(cesd, xl, cesi, xl_file);
#pragma omp critical
    crcs[idx] ^= crc;
  }
  if (cesb.geo.geon.size_node == cesi.total_node) {
    const int id_node = cesb.geo.geon.id_node;
    if (crcs[id_node] == cesi.crcs[id_node]) {
      displayln(fname + ssprintf(": crc check successfull."));
    } else {
      displayln(fname + ssprintf(": ERROR: crc check failed id_node=%d read=%08X computed=%08X.",
            id_node, cesi.crcs[id_node], crcs[id_node]));
      ssleep(1.0);
      qassert(false);
    }
  }
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    load_block(cesb, cesc, cesd, xl, cesi);
  }
  return crcs;
}

struct BlockedHalfVector : Field<ComplexF>
{
  long block_vol_eo; // even odd precondition (block_vol is half of the number of site within the block)
  int ls;
  // geo.multiplicity = block_vol_eo * ls * HalfVector::c_size;
  Geometry geo_full;
  Coordinate block_site;
  //
  virtual const std::string& cname()
  {
    static const std::string s = "BlockedHalfVector";
    return s;
  }
};

inline void init_blocked_half_vector(
    BlockedHalfVector& bhv,
    const Geometry& geo_full,
    const Coordinate& block_site,
    const int ls)
{
  TIMER("init_blocked_half_vector");
  bhv.block_vol_eo = product(block_site) / 2;
  bhv.ls = ls;
  const Geometry geo = geo_remult(block_geometry(geo_full, block_site), bhv.block_vol_eo * ls * HalfVector::c_size);
  bhv.geo_full = geo_full;
  bhv.block_site = block_site;
  bhv.init();
  bhv.init(geo);
}

inline void decompress_eigen_system(
    std::vector<BlockedHalfVector>& bhvs,
    const CompressedEigenSystemBases& cesb,
    const CompressedEigenSystemCoefs& cesc)
{
  TIMER_VERBOSE("decompress_eigen_system");
  const Geometry geo_full = geo_reform(cesb.geo_full);
  const Coordinate& block_site = cesb.block_site;
  const int ls = cesb.ls;
  qassert(geo_remult(cesb.geo) == geo_remult(cesc.geo));
  qassert(geo_full == cesb.geo_full);
  qassert(geo_full == cesc.geo_full);
  qassert(block_site == cesc.block_site);
  qassert(ls == cesc.ls);
  const int n_vec = cesc.n_vec;
  if (n_vec == 0) {
    return;
  }
  {
    TIMER_VERBOSE("decompress_eigen_system-init");
    bhvs.resize(n_vec);
    for (int i = 0; i < n_vec; ++i) {
      init_blocked_half_vector(bhvs[i], geo_full, block_site, ls);
    }
  }
  const long block_size = cesb.block_vol_eo * ls * HalfVector::c_size;
  const long n_basis = cesb.n_basis;
  qassert(n_basis == cesc.n_basis);
  const Geometry& geo = bhvs[0].geo;
  qassert(block_size == geo.multiplicity);
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Vector<ComplexF> bases = cesb.get_elems_const(index);
    const Vector<ComplexF> coefs = cesc.get_elems_const(index);
    std::vector<Vector<ComplexF> > blocks(n_vec);
    for (int i = 0; i < n_vec; ++i) {
      blocks[i] = bhvs[i].get_elems(index);
    }
    TIMER_VERBOSE_FLOPS("decompress_eigen_system-block");
    timer.flops += n_vec * n_basis * block_size * 8;
#pragma omp parallel for
    for (int i = 0; i < n_vec; ++i) {
      for (int j = 0; j < n_basis; ++j) {
        caxpy_single(blocks[i].data(), coefs[i * n_basis + j], &bases[j * block_size], blocks[i].data(), block_size);
      }
    }
  }
}

inline void convert_half_vector(HalfVector& hv, const BlockedHalfVector& bhv)
{
  TIMER("convert_half_vector");
  const Coordinate& block_site = bhv.block_site;
  const int ls = bhv.ls;
  init_half_vector(hv, bhv.geo_full, ls);
  const Geometry& geo = hv.geo;
  const Coordinate node_block = geo.node_site / block_site;
  qassert(geo.is_only_local());
#pragma omp parallel for
  for (long index = 0; index < geo.local_volume(); ++index) {
    const Coordinate xl = geo.coordinate_from_index(index);
    qassert(xl[0] % 2 == 1);
    const Coordinate bxl = xl / block_site;
    const long bindex = index_from_coordinate(bxl, node_block);
    Vector<ComplexF> site = hv.get_elems(index);
    qassert(site.size() == ls * HalfVector::c_size);
    const long bidx = index_from_coordinate(xl % block_site, block_site)/2;
    memcpy(
        site.data(),
        &bhv.get_elems_const(bindex)[bidx * site.size()],
        site.data_size());
  }
}

inline void convert_half_vectors(std::vector<HalfVector>& hvs, std::vector<BlockedHalfVector>& bhvs)
  // will clear bhvs to save space
{
  TIMER_VERBOSE("convert_half_vectors");
  hvs.clear();
  hvs.resize(bhvs.size());
  for (int i = 0; i < (int)hvs.size(); ++i) {
    convert_half_vector(hvs[i], bhvs[i]);
    bhvs[i].init();
  }
  bhvs.clear();
}

inline void save_half_vectors(const std::vector<HalfVector>& hvs, const std::string& fn, const bool is_bfm_format = false)
{
  TIMER_VERBOSE_FLOPS("save_half_vectors");
  qassert(hvs.size() > 0);
  timer.flops += get_data(hvs[0]).data_size() * hvs.size();
  const long size = hvs[0].field.size();
  std::vector<ComplexF> buffer(size);
  crc32_t crc = 0;
  FILE* fp = qopen(fn + ".partial", "w");
  qassert(fp != NULL);
  for (int i = 0; i < (int)hvs.size(); ++i) {
    TIMER_FLOPS("save_half_vectors-iter");
    timer.flops += get_data(hvs[i]).data_size();
    qassert(hvs[i].geo.is_only_local());
    if (is_bfm_format) {
      for (long m = 0; m < size/2; ++m) {
        buffer[m*2] = hvs[i].field[m];
        buffer[m*2+1] = hvs[i].field[size/2+m];
      }
    } else {
      buffer = hvs[i].field;
    }
    to_from_big_endian_32(get_data(buffer));
    crc = crc32_par(crc, get_data(buffer));
    qwrite_data(get_data(buffer), fp);
  }
  qclose(fp);
  qtouch(fn + ".crc32", ssprintf("%08X\n", crc));
  qrename(fn + ".partial", fn);
}

inline long decompress_eigen_vectors_block(
    const std::string& old_path, const CompressedEigenSystemInfo& cesi,
    const std::string& new_path, const int idx, const Coordinate& new_size_node = Coordinate())
  // interface
  // new_size_node can be Coordinate()
  // single node code
{
  TIMER_VERBOSE("decompress_eigen_vectors_block");
  Coordinate size_node = new_size_node;
  if (size_node == Coordinate()) {
    size_node = cesi.total_node;
  }
  const int idx_size = product(size_node);
  const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
  const std::string data_path = old_path + ssprintf("/%02d/%010d", dir_idx, idx);
  qmkdir(new_path);
  qmkdir(new_path + ssprintf("/%02d", dir_idx));
  const std::string new_fn = new_path + ssprintf("/%02d/%010d", dir_idx, idx);
  CompressedEigenSystemBases cesb;
  CompressedEigenSystemCoefs cesc;
  init_compressed_eigen_system_bases(cesb, cesi, idx, size_node);
  init_compressed_eigen_system_coefs(cesc, cesi, idx, size_node);
  std::vector<crc32_t> crcs = load_node(cesb, cesc, cesi, old_path);
  to_from_big_endian_32(get_data(crcs));
  FILE* fp = qopen(new_fn + ".orig-crc32", "w");
  qassert(fp != NULL);
  qwrite_data(get_data(crcs), fp);
  qclose(fp);
  std::vector<BlockedHalfVector> bhvs;
  decompress_eigen_system(bhvs, cesb, cesc);
  std::vector<HalfVector> hvs;
  convert_half_vectors(hvs, bhvs);
  save_half_vectors(hvs, new_fn, true);
  return 0;
}

inline long decompress_eigen_vectors_block(
    const std::string& old_path,
    const std::string& new_path, const int idx, const Coordinate& new_size_node = Coordinate())
  // interface
  // new_size_node can be Coordinate()
  // single node code
{
  const CompressedEigenSystemInfo cesi = read_compressed_eigen_system_info(old_path);
  return decompress_eigen_vectors_block(old_path, cesi, new_path, idx, new_size_node);
}

inline void set_lock_expiration_time_limit()
{
  const std::string ss = get_env("COBALT_STARTTIME");
  const std::string se = get_env("COBALT_ENDTIME");
  if (ss != "" and se != "") {
    TIMER_VERBOSE("set_lock_expiration_time_limit");
    double start_time, end_time;
    reads(start_time, ss);
    reads(end_time, se);
    get_lock_expiration_time_limit() = end_time - start_time;
  }
}

inline crc32_t compute_crc32(const std::string& path)
{
  TIMER_VERBOSE_FLOPS("compute_crc32");
  const size_t chunk_size = 16 * 1024 * 1024;
  std::vector<char> data(chunk_size);
  crc32_t crc = 0;
  FILE* fp = qopen(path, "r");
  qassert(fp != NULL);
  while (true) {
    const long size = qread_data(get_data(data), fp);
    timer.flops += size;
    if (size == 0) {
      break;
    }
    crc = crc32_par(crc, Vector<char>(data.data(), size));
  }
  qclose(fp);
  return crc;
}

inline void combine_crc32(const std::string& path, const int idx_size, const CompressedEigenSystemInfo& cesi)
{
  TIMER_VERBOSE("combine_crc32");
  if (0 == get_id_node()) {
    {
      // check orig-crc32
      FILE* mfp = qopen(path + "/combine_crc32.log", "a");
      qassert(mfp != NULL);
      qset_line_buf(mfp);
      get_monitor_file() = mfp;
      const std::vector<crc32_t>& crcs_orig = cesi.crcs;
      const int size = crcs_orig.size();
      std::vector<crc32_t> crcs_acc(size, 0);
      for (int idx = 0; idx < idx_size; ++idx) {
        std::vector<crc32_t> crcs(size, 0);
        const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
        FILE* fp = qopen(path + ssprintf("/%02d/%010d.orig-crc32", dir_idx, idx), "r");
        if (fp != NULL) {
          qread_data(get_data(crcs), fp);
          to_from_big_endian_32(get_data(crcs));
          qclose(fp);
        } else {
          displayln(fname + ssprintf(": ERROR: %02d/%010d orig-crc32 do not exist", dir_idx, idx));
          continue;
        }
        for (int i = 0; i < size; ++i) {
          crcs_acc[i] ^= crcs[i];
        }
      }
      for (int i = 0; i < size; ++i) {
        if (crcs_acc[i] != crcs_orig[i]) {
          displayln(fname + ssprintf(": ERROR: mismatch %d/%d orig=%08X acc=%08X", i, size, crcs_orig[i], crcs_acc[i]));
        } else {
          displayln(fname + ssprintf(": match %d/%d orig=%08X acc=%08X", i, size, crcs_orig[i], crcs_acc[i]));
        }
      }
      get_monitor_file() = NULL;
      qclose(mfp);
    }
    const std::string fn = path + "/checksums.txt";
    if (not does_file_exist(fn)) {
      std::vector<crc32_t> crcs(idx_size, 0);
      for (int idx = 0; idx < idx_size; ++idx) {
        const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
        const std::string path_data = path + ssprintf("/%02d/%010d", dir_idx, idx);
        if (does_file_exist(path_data + ".crc32")) {
          crcs[idx] = read_crc32(qcat(path_data + ".crc32"));
        } else {
          crcs[idx] = compute_crc32(path_data);
          qtouch(path_data + ".crc32", ssprintf("%08X\n", crcs[idx]));
        }
        displayln(ssprintf("reading crc %7d/%d %08X", idx, idx_size, crcs[idx]));
      }
      crc32_t crc = dist_crc32(crcs);
      FILE* fp = qopen(fn + ".partial", "w");
      qassert(fp != NULL);
      displayln(ssprintf("%08X", crc), fp);
      displayln("", fp);
      for (size_t i = 0; i < crcs.size(); ++i) {
        displayln(ssprintf("%08X", crcs[i]), fp);
      }
      qclose(fp);
      qrename(fn + ".partial", fn);
    }
    qtouch(path + "/checkpoint");
    // for (int idx = 0; idx < idx_size; ++idx) {
    //   const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
    //   qremove(path + ssprintf("/%02d/%010d.crc32", dir_idx, idx));
    //   qremove(path + ssprintf("/%02d/%010d.orig-crc32", dir_idx, idx));
    // }
  }
}

inline void decompress_eigen_vectors(const std::string& old_path, const std::string& new_path, const Coordinate& new_size_node = Coordinate())
  // interface
{
  if (does_file_exist_sync_node(new_path + "/checkpoint")) {
    return;
  }
  TIMER_VERBOSE("decompress_eigen_vectors");
  displayln_info(fname + ssprintf(": old_path: '") + old_path + "'");
  displayln_info(fname + ssprintf(": new_path: '") + new_path + "'");
  qmkdir_info(new_path);
  set_lock_expiration_time_limit();
  CompressedEigenSystemInfo cesi;
  cesi = read_compressed_eigen_system_info(old_path);
  Coordinate size_node = new_size_node;
  if (size_node == Coordinate()) {
    size_node = cesi.total_node;
  }
  long idx_size = product(size_node);
  if (0 == get_id_node()) {
    const std::string eigen_values = qcat(old_path + "/eigen-values.txt");
    qtouch(new_path + "/eigen-values.txt", eigen_values);
  }
  displayln_info(fname + ssprintf(": idx_size=%d", idx_size));
  std::vector<int> avails(idx_size, 0);
  if (0 == get_id_node()) {
    for (int idx = 0; idx < idx_size; ++idx) {
      const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
      if (!does_file_exist(new_path + ssprintf("/%02d/%010d", dir_idx, idx))) {
        avails[idx] = 1;
      }
    }
  }
  bcast(get_data(avails));
  int n_avails = 0;
  for (int idx = 0; idx < idx_size; ++idx) {
    if (avails[idx] != 0) {
      n_avails += 1;
    }
  }
  displayln_info(fname + ssprintf(": n_avails=%d", n_avails));
  if (n_avails == 0) {
    if (obtain_lock(new_path + "/lock")) {
      combine_crc32(new_path, idx_size, cesi);
      release_lock();
    }
    return;
  }
  std::vector<int> order(n_avails, -1);
  {
    TIMER_VERBOSE("decompress_eigen_vectors-shuffle");
    RngState rs(get_global_rng_state(), fname);
    split_rng_state(rs, rs, show(get_time()));
    split_rng_state(rs, rs, get_id_node());
    for (int i = 0; i < n_avails; ++i) {
      int idx = -1;
      do {
        idx = rand_gen(rs) % idx_size;
      } while (avails[idx] == 0);
      avails[idx] = 0;
      order[i] = idx;
    }
  }
  ssleep(2.0*get_id_node());
  int num_done = 0;
  for (int i = 0; i < (int)order.size(); ++i) {
    displayln(fname + ssprintf(": %5d/%d order[%03d/%05d/%d]=%010d", get_id_node(), get_num_node(), num_done, i, order.size(), order[i]));
    const int idx = order[i];
    const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
    const std::string path_data = new_path + ssprintf("/%02d/%010d", dir_idx, idx);
    const std::string path_lock = new_path + ssprintf("/%02d-%010d-lock", dir_idx, idx);
    if (!does_file_exist(path_data) and obtain_lock_all_node(path_lock)) {
      decompress_eigen_vectors_block(old_path, cesi, new_path, idx, new_size_node);
      release_lock_all_node();
      num_done += 1;
      displayln(fname + ssprintf(": %5d/%d order[%03d/%05d/%d]=%010d finished", get_id_node(), get_num_node(), num_done, i, order.size(), order[i]));
      Timer::display();
    }
  }
  displayln(fname + ssprintf(": process %5d/%d finish '", get_id_node(), get_num_node()) + old_path + "'");
}

inline void decompressed_eigen_vectors_check_crc32(const std::string& path)
  // interface
{
  if (does_file_exist_sync_node(path + "/checksums-check.txt") and does_file_exist_sync_node(path + "/checkpoint")) {
    return;
  }
  TIMER_VERBOSE("decompressed_eigen_vectors_check_crc32");
  if (not obtain_lock(path + "/lock")) {
    return;
  }
  displayln_info(ssprintf("%s: ", fname) + path);
  qassert(does_file_exist_sync_node(path + "/checksums.txt"));
  set_lock_expiration_time_limit();
  long idx_size = 0;
  std::vector<crc32_t> crcs_load;
  if (get_id_node() == 0) {
    const std::string fn = path + "/checksums.txt";
    const std::vector<std::string> lines = qgetlines(fn);
    for (size_t i = 0; i < lines.size() - 2; ++i) {
      const std::string& l = lines[i + 2];
      if (l.size() > 0 and l[0] != '\n') {
        crcs_load.push_back(read_crc32(l));
      }
    }
    idx_size = crcs_load.size();
    displayln(fname + ssprintf(": idx_size=%d", idx_size));
    glb_sum(idx_size);
    const crc32_t crc_l0 = read_crc32(lines[0]);
    const crc32_t crc_c0 = dist_crc32(crcs_load);
    displayln(ssprintf("summary crc stored=%08X computed=%08X", crc_l0, crc_c0));
    qassert(crc_l0 == crc_c0);
  } else {
    glb_sum(idx_size);
    crcs_load.resize(idx_size);
  }
  bcast(get_data(crcs_load));
  std::vector<crc32_t> crcs(idx_size, 0);
// #pragma omp parallel for
  for (int idx = get_id_node(); idx < idx_size; idx += get_num_node()) {
    const int dir_idx = compute_dist_file_dir_id(idx, idx_size);
    const std::string path_data = path + ssprintf("/%02d/%010d", dir_idx, idx);
    crcs[idx] = compute_crc32(path_data);
    displayln(ssprintf("reading crc %7d/%d stored=%08X computed=%08X", idx, idx_size, crcs_load[idx], crcs[idx]));
    if (crcs_load[idx] != crcs[idx]) {
      displayln(ssprintf("reading crc %7d/%d stored=%08X computed=%08X mismatch ('%s')", idx, idx_size, crcs_load[idx], crcs[idx], path.c_str()));
      qremove(path_data);
      qremove(path_data + ".crc32");
      qremove(path + "/checkpoint");
    }
  }
  glb_sum_byte_vec(get_data(crcs));
  if (0 == get_id_node()) {
    crc32_t crc = dist_crc32(crcs);
    const std::string fn = path + "/checksums-check.txt";
    FILE* fp = qopen(fn + ".partial", "w");
    qassert(fp != NULL);
    displayln(ssprintf("%08X", crc), fp);
    displayln("", fp);
    for (size_t i = 0; i < crcs.size(); ++i) {
      displayln(ssprintf("%08X", crcs[i]), fp);
    }
    qclose(fp);
    qrename(fn + ".partial", fn);
  }
  release_lock();
}

inline int count_64I_todo(const int traj)
{
  TIMER("count_64I_todo");
  int count = 2;
  if (does_file_exist_sync_node(ssprintf("/home/ljin/application/Public/Muon-GM2-cc/jobs/64I/dis-1/results/results=%d/checkpoint/computeContraction", traj/10))) {
    count -= 1;
  }
  if (does_file_exist_sync_node(ssprintf("/home/ljin/application/Public/Muon-GM2-cc/jobs/64I/2/results/results=%d/checkpoint", traj/10))) {
    count -= 1;
  }
  displayln_info(fname + ssprintf(": traj=%d todo=%d", traj, count));
  return count;
}

inline void decompress_64I()
{
  TIMER_VERBOSE("decompress_64I");
  get_lock_expiration_time_limit() = 3600.0;
  std::vector<int> trajs;
  {
    const int skip = 40;
    int limit = 1300;
    for (int traj = 1300; traj < 9990; traj += 10) {
      if (count_64I_todo(traj) <= 1) {
        limit = traj + skip;
      } else if (traj >= limit) {
        const std::string root = ssprintf("/home/ljin/hlbl/clehner/evec-cache/64I_mg/job-%05d/lanczos.output", traj);
        if (does_file_exist_sync_node(root + "/metadata.txt")) {
          trajs.push_back(traj);
          displayln_info(ssprintf("add %d", traj));
          limit = traj + skip;
        }
      }
    }
  }
  {
    const int skip = 20;
    int limit = 1300;
    for (int traj = 1300; traj < 9990; traj += 10) {
      if (count_64I_todo(traj) <= 1) {
        limit = traj + skip;
      } else if (traj >= limit) {
        const std::string root = ssprintf("/home/ljin/hlbl/clehner/evec-cache/64I_mg/job-%05d/lanczos.output", traj);
        if (does_file_exist_sync_node(root + "/metadata.txt")) {
          trajs.push_back(traj);
          displayln_info(ssprintf("add backup %d", traj));
          limit = traj + skip;
        }
      }
    }
  }
  for (int traj = 1300; traj < 9990; traj += 10) {
    trajs.push_back(traj);
  }
  for (long i = 0; i < (long)trajs.size(); ++i) {
    const int traj = trajs[i];
    if (count_64I_todo(traj) == 0) {
      continue;
    }
    const std::string root = ssprintf("/home/ljin/hlbl/clehner/evec-cache/64I_mg/job-%05d/lanczos.output", traj);
    if (!does_file_exist_sync_node(root + "/metadata.txt")) {
      continue;
    }
    const std::string new_root = ssprintf(
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/64I/clehner/regular-eigen-vectors/qcdtraj=%d", traj);
    qmkdir_info(new_root);
    decompress_eigen_vectors(root, new_root + "/huge-data-lanc", Coordinate(4, 8, 16, 16));
  }
}

inline void check_checksum_64I()
{
  get_lock_expiration_time_limit() = 3600.0;
  for (int traj = 1290; traj < 9990; traj += 10) {
    const std::string root = ssprintf(
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/64I/clehner/regular-eigen-vectors/qcdtraj=%d", traj);
    if (!does_file_exist_sync_node(root + "/huge-data-lanc/checksums.txt")) {
      continue;
    }
    decompressed_eigen_vectors_check_crc32(root + "/huge-data-lanc");
  }
}

inline void decompress_48D()
{
  TIMER_VERBOSE("decompress_48D");
  get_lock_expiration_time_limit() = 3600.0;
  std::vector<int> trajs;
  trajs.push_back(490);
  for (long i = 0; i < (long)trajs.size(); ++i) {
    const int traj = trajs[i];
    const std::string root = ssprintf("/home/ljin/hlbl/clehner/evec-cache/48D/job-%05d/lanczos.output", traj);
    if (!does_file_exist_sync_node(root + "/metadata.txt")) {
      continue;
    }
    const std::string new_root = ssprintf(
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/48D/clehner/regular-eigen-vectors/qcdtraj=%d", traj);
    qmkdir_info(new_root);
    decompress_eigen_vectors(root, new_root + "/huge-data-lanc", Coordinate(8, 4, 8, 16));
  }
}

inline void check_checksum_48D()
{
  get_lock_expiration_time_limit() = 3600.0;
  for (int traj = 100; traj < 9990; traj += 10) {
    const std::string root = ssprintf(
        "/home/ljin/application/Public/Muon-GM2-cc/jobs/48D/clehner/regular-eigen-vectors/qcdtraj=%d", traj);
    if (!does_file_exist_sync_node(root + "/huge-data-lanc/checksums.txt")) {
      continue;
    }
    decompressed_eigen_vectors_check_crc32(root + "/huge-data-lanc");
  }
}

inline void test_decompress_eigen_vectors_block()
{
  const std::string root = get_env("HOME") + "/application/Public/Muon-GM2-cc/jobs/64I/clehner/compressed-eigen-vectors/ckpoint_lat.1320.evecs";
  const CompressedEigenSystemInfo cesi = read_compressed_eigen_system_info(root);
  const int idx = 3023;
  const int dir_idx = compute_dist_file_dir_id(idx, product(cesi.total_node));
  decompress_eigen_vectors_block(root, cesi, "huge-data", idx, Coordinate());
  displayln_info("old crc32: 8DBFFD60\nnew crc32: " + qcat_info(ssprintf("huge-data/%02d/%010d.crc32", dir_idx, idx)));
}

QLAT_END_NAMESPACE

//int main(int argc, char* argv[])
//{
//  using namespace qlat;
//  begin(&argc, &argv);
//  if (0 == get_id_node()) {
//    displayln_info(ssprintf("OMP_NUM_THREADS=%d", omp_get_max_threads()));
//  }
//  // ADJUST ME
//  check_checksum_64I();
//  decompress_64I();
//  // decompress_48D();
//  // check_checksum_48D();
//  //
//  // test_decompress_eigen_vectors_block();
//  // decompressed_eigen_vectors_check_crc32("/home/ljin/hlbl/chulwoo/64I_gfixed/results-256/huge-data-lanc");
//  //
//  // decompress_eigen_vectors_block("/home/ljin/hlbl/clehner/evec-cache/64I_mg/job-01220/lanczos.output", "huge-data", 1220, Coordinate(4, 8, 16, 16));
//  // decompress_eigen_vectors_block("/home/ljin/hlbl/clehner/evec-cache/64I/ckpoint_lat.2700.evecs", "/home/ljin/application/Public/Muon-GM2-cc/jobs/64I/clehner/regular-eigen-vectors/qcdtraj=2700/huge-data-lanc", 6962);
//  //
//  Timer::display();
//  end();
//  return 0;
//}
