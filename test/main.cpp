
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
        // int ncomp = 3; 
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
        amrex::MultiFab rhs(ba, dm, 1, 4);
        amrex::MultiFab phi_old(ba, dm, 2, 4);
        
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
            if( 7 < phi_old_array(i,j,k,0) && phi_old_array(i,j,k,0)< 9 )
            { rhs_array(i,j,k) = 9999;//i = 8
              rhs_array(7,j,k) = 8888;//i-1 = 7 
              rhs_array(i+1,j,k) = 10000;//i+1 = 9
	      rhs_array(i-2,j,k) = 7777 ;//i-2 = 6
              rhs_array(i+2,j,k) = 11000;//i+2 = 10
             //
	     //
	     //rhs_array(i-4,j,k) = 5555;
             // rhs_array(i-5,j,k) = 6666;
	    }
        });
    };
        
        WriteSingleLevelPlotfile("phi_old", phi_old, {"phi_old","2000"}, geom, 0., 0);
        WriteSingleLevelPlotfile("rhs", rhs, {"rhs"}, geom, 0., 0);
    
    }
    amrex::Finalize();
}
