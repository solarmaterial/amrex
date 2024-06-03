
#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_MFParallelFor.H> //For the second newer method
#include <AMReX_PlotFileUtil.H> //For ploting the MultiFab
#include <AMReX_MultiFabUtil.H>
using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";


        // Goals:
        // Define a MultiFab
        // Fill a MultiFab with data
        // Plot it


        // Parameters

        // Number of data components at each grid point in the MultiFab
        int ncomp = 1;
        // how many grid cells in each direction over the problem domain
        int n_cell = 32;
        // how many grid cells are allowed in each direction over each box
        int max_grid_size = 8;

        //BoxArray -- Abstract Domain Setup


        // integer vector indicating the lower coordindate bounds
        amrex::IntVect dom_lo(0,0,0);
        // integer vector indicating the upper coordindate bounds
        amrex::IntVect dom_hi(n_cell-1, n_cell-1, n_cell-1);
        // box containing the coordinates of this domain
        amrex::Box domain(dom_lo, dom_hi);

        // will contain a list of boxes describing the problem domain
        amrex::BoxArray ba(domain);

        // chop the single grid into many small boxes
        ba.maxSize(max_grid_size);

        // Distribution Mapping
        amrex::DistributionMapping dm(ba);

        //Define MuliFab
        amrex::MultiFab rhs(ba, dm, 1, 0);
        amrex::MultiFab phi_old(ba, dm, 2, 0);

        //Geometry -- Physical Properties for data on our domain
        amrex::RealBox real_box ({0., 0., 0.}, {1. , 1., 1.});

        amrex::Geometry geom(domain, &real_box);


        //Calculate Cell Sizes
        amrex::GpuArray<amrex::Real,3> dx = geom.CellSizeArray();  //dx[0] = dx dx[1] = dy dx[2] = dz



        for(amrex::MFIter mfi(phi_old); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& phi_old_array = phi_old.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
                phi_old_array(i,j,k,0) = i;
                phi_old_array(i,j,k,1) = 2000;
            });
        }

        rhs.setVal(0.0);
        for(amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi){
            const amrex::Box& bx = mfi.validbox();
            const amrex::Array4<amrex::Real>& rhs_array = rhs.array(mfi);
            const amrex::Array4<amrex::Real>& phi_old_array = phi_old.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
                
                int i0, i1;

                if( phi_old_array(i,j,k,0) - 17. == 0. )
                {
                    i1 = i;
                    i0 = i - 5;       
                }
                if(i0 < i && i <= i1)
                {
                    rhs_array(i,j,k) = 666;
                }

            });
        }
        WriteSingleLevelPlotfile("phi_old", phi_old, {"i","2000"}, geom, 0., 0);
        WriteSingleLevelPlotfile("rhs", rhs, {"666"}, geom, 0., 0);
    
    }
    amrex::Finalize();
}
