// -*- mode:c++; c-basic-offset:4 -*-
#ifndef EIGCG_H_KL3__
#define EIGCG_H_KL3__

#include <alg/eigcg_arg.h>

class EigCG {
public:
    EigCG(cps::EigCGArg *eigcg_arg, int Ls, bool use_float = false, int nx=50);
    ~EigCG();

    void printH(const std::string &fn)const;
    void printG(const std::string &fn)const;
	int get_num_low_modes_collected() const;
private:
    bool created;
    bool is_float;
};

#endif
