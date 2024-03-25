
#include "myfunc.H"
#include "myfunc_F.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
using namespace amrex;

void advance (MultiFab& phi_old,
              MultiFab& phi_new,
              Real dt,
              const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc)
{
    /*
      We use an MLABecLaplacian operator:

      (ascalar*acoef - bscalar div bcoef grad) phi = RHS

      for an implicit discretization of the *nonlinearized* heat equation

      (I - dt div beta(phi^{n+1}) grad) phi^{n+1} = phi^n
    */

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries，
    phi_old.FillBoundary(geom.periodicity());

    // Fill non-periodic physical boundaries
    FillDomainBoundary(phi_old, geom, bc);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;

    for (int n = 0; n < phi_old.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            // lo-side BCs
            if (bc[n].lo(idim) == BCType::int_dir) {
                bc_lo[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].lo(idim) == BCType::foextrap) {
                bc_lo[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].lo(idim) == BCType::ext_dir) {
                bc_lo[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc[n].hi(idim) == BCType::int_dir) {
                bc_hi[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].hi(idim) == BCType::foextrap) {
                bc_hi[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].hi(idim) == BCType::ext_dir) {
                bc_hi[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(bc_lo, bc_hi);

    // set the boundary conditions
    mlabec.setLevelBC(0, &phi_old);

    // scaling factors,
    Real ascalar = 1.0;
    Real bscalar = dt;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices,BoxArray grids,DistributionMapping dmap
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(1.0);
    mlabec.setACoeffs(0, acoef);

    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
       std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
    }

    
    MultiFab phi_k(grids, dmap, 1, 1);//β(phi_k)
    MultiFab phi_iter_solve(grids, dmap, 1, 1);//compute phi_(s+1)
    MultiFab phi_iter_old(grids, dmap, 1, 1);//store the data of phi_(s)
    MultiFab phi_res(grids, dmap, 1, 1);//phi_res = phi_iter_solve - phi_iter_old

    MultiFab::Copy(phi_k, phi_old, 0, 0, 1, 1);//β(phi_k) is loaded
    MultiFab::Copy(phi_iter_old, phi_old, 0, 0, 1, 1);//

    const Real sigma = 1.e-6;//|phi_iter_solve - phi_iter_old|<sigma; break
    bool should_break = false;//break when `true`

    for (int n = 0; n < 1.e8; ++n)//non-linear iteration starts here
{
    Multiply (phi_k, phi_k, 0, 0, 1, 1);//β(phi_k) = (phi_k)^2
    WriteSingleLevelPlotfile("phi_k", phi_k, {"comp0"}, geom, 0., 0);


/////////////////////////
   for ( amrex::MFIter mfi(phi_k); mfi.isValid(); ++mfi )
    {
    const Box& xbx = mfi.nodaltilebox(0);
    auto const& face_bcoef_x = face_bcoef[0].array(mfi);
    Dim3 lox = lbound(xbx); Dim3 hix = ubound(xbx);

#if (AMREX_SPACEDIM > 1)    
    const Box& ybx = mfi.nodaltilebox(1);
    auto const& face_bcoef_y = face_bcoef[1].array(mfi);
    Dim3 loy = lbound(ybx); Dim3 hiy = ubound(ybx);
#endif

#if (AMREX_SPACEDIM > 2)
    const Box& zbx = mfi.nodaltilebox(2);
    auto const& face_bcoef_z = face_bcoef[2].array(mfi);
    Dim3 loz = lbound(zbx); Dim3 hiz = ubound(zbx);
#endif
    
    auto const& array_phi_k = phi_k.array(mfi);

        amrex::ParallelFor(xbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if(i == hix.x)
            {face_bcoef_x(i,j,k) = face_bcoef_x(i-1,j,k);}
            else if(i == lox.x)
            {face_bcoef_x(i,j,k) = (array_phi_k(i,j,k) + array_phi_k(i+1,j,k)) / 2;}
            else
            {face_bcoef_x(i,j,k) = (array_phi_k(i-1,j,k) + array_phi_k(i,j,k)) / 2;}
        });

#if (AMREX_SPACEDIM > 1)
        amrex::ParallelFor(ybx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if(j == hiy.y)
            {face_bcoef_y(i,j,k) = face_bcoef_y(i,j-1,k);}
            else if(j == loy.y)
            {face_bcoef_y(i,j,k) = (array_phi_k(i,j,k) + array_phi_k(i,j+1,k)) / 2;}
            else
            {face_bcoef_y(i,j,k) = (array_phi_k(i,j-1,k) + array_phi_k(i,j,k)) / 2;}
        });
#endif

#if (AMREX_SPACEDIM > 2)
        amrex::ParallelFor(zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if(k == hiz.z)
            {face_bcoef_z(i,j,k) = face_bcoef_z(i,j,k-1);}
            else if(k == loz.z)
            {face_bcoef_z(i,j,k) = (array_phi_k(i,j,k) + array_phi_k(i,j,k+1)) / 2;}
            else
            {face_bcoef_z(i,j,k) = (array_phi_k(i,j,k-1) + array_phi_k(i,j,k)) / 2;}
        });
#endif
    }
////////////////////////////


    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));
    //WriteSingleLevelPlotfile("face_x0", face_bcoef[0], {"comp0"}, geom, 0., 0);

    // build an MLMG solver
    MLMG mlmg(mlabec);
    //mlmg.setBottomSolver (BottomSolver::cgbicg);
    
    // set solver parameters
    int max_iter = 100;
    mlmg.setMaxIter(max_iter);
    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);
    int verbose = 2;
    mlmg.setVerbose(verbose);
    int bottom_verbose = 1;
    mlmg.setBottomVerbose(bottom_verbose);


    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;

    // Solve linear system
    mlmg.solve({&phi_iter_solve}, {&phi_old}, tol_rel, tol_abs);//phi_old remain unchanged during the non-linear iteration

    
    //WriteSingleLevelPlotfile("phi_iter_solve", phi_iter_solve, {"comp0"}, geom, 0., 0);
    //WriteSingleLevelPlotfile("phi_old", phi_old, {"comp0"}, geom, 0., 0);

    //compute phi_res = phi_iter_old + (-1)phi_iter_solve
    MultiFab::LinComb(phi_res, 1.0, phi_iter_solve, 0, -1.0, phi_iter_old, 0, 0, 1, 0);
    
    //WriteSingleLevelPlotfile("phi_res", phi_res, {"comp0"}, geom, 0., 0);
   
    Real maxnorm_phi_res = phi_res.norm0 (0, 0, 0, 0);//is ghost cell needed? 
    //amrex::Print() << "maxnorm_phi_res = " << maxnorm_phi_res << "\n";  
   
    if(maxnorm_phi_res < sigma)
   {
    should_break = true;
    WriteSingleLevelPlotfile("face_x", face_bcoef[0], {"comp0"}, geom, 0., 0);
    amrex::Print() << "fine_phi_res = " << maxnorm_phi_res << "\n";   
    break;
   }
    
    MultiFab::Copy(phi_k, phi_iter_solve, 0, 0, 1, 1); //copy phi_iter_solve to phi_k to start next iteration
    MultiFab::Copy(phi_iter_old, phi_iter_solve, 0, 0, 1, 1);//copy phi_iter_solve(phi_(s+1)) to phi_iter_old(phi_(s))
}
      
    if (should_break) 
    {  
    MultiFab::Copy(phi_new, phi_iter_solve, 0, 0, 1, 1); // copy phi_iter_solve to phi_new, transfer phi_new to main.cpp. Advance time step.
    }   

}
