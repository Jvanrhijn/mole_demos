use ndarray::{array, Array2};
use mole::{
    metropolis::{
        MetropolisDiffuse,
        MetropolisBox,
    },
    wavefunction::{
        Orbital, 
        SpinDeterminantProduct, 
        SingleDeterminant,
        Error,
        Cache,
    },
    montecarlo::{Runner, Sampler},
    basis::Hydrogen1sBasis,
    operator::{Operator}
};
use rand::{SeedableRng, StdRng};

struct Coordinate { 
    coordinate_idx: usize, 
    electron: usize 
}

impl Coordinate {
    pub fn new(coordinate_idx: usize, electron: usize) -> Self {
        Self {coordinate_idx, electron}
    }
}

impl<T: Cache> Operator<T> for Coordinate {
    fn act_on(&self, wf: &T, cfg: &Array2<f64>) -> Result<f64, Error> {
        Ok(cfg[[self.electron, self.coordinate_idx]] * wf.current_value().0)
    }
}

fn main() {
    let ion_pos = array![
        [-0.7, 0.0, 0.0], 
        [0.7, 0.0, 0.0],
    ];

    let basis = Hydrogen1sBasis::new(ion_pos.clone(), vec![1.0]);

    let orbitals = vec![
        Orbital::new(array![[1.0], [0.0]], basis.clone()),
        Orbital::new(array![[0.0], [1.0]], basis.clone()),
    ];

    let wave_function = SpinDeterminantProduct::new(orbitals, 1);

    let rng = StdRng::from_seed([0u8; 32]);
    let metrop = MetropolisDiffuse::from_rng(0.001, rng);

    let mut sampler = Sampler::new(wave_function, metrop);
    sampler.add_observable("x1", Coordinate::new(0, 0));
    sampler.add_observable("y1", Coordinate::new(1, 0));
    sampler.add_observable("z1", Coordinate::new(2, 0));
    sampler.add_observable("x2", Coordinate::new(0, 1));
    sampler.add_observable("y2", Coordinate::new(1, 1));
    sampler.add_observable("z2", Coordinate::new(2, 1));

    Runner::new(sampler)
        .run(1_000_000, 100);
}
