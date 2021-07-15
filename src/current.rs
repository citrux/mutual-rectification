extern crate tini;
extern crate time;
extern crate scoped_threadpool;
extern crate scattering;
extern crate linal;

mod material;

use std::io::{stdin, Read};

use tini::Ini;
use time::{get_time, SteadyTime};
use scoped_threadpool::Pool;
use scattering::particle::Summary;
use scattering::{Fields, Stats, create_ensemble};
use material::SL3;

fn main() {
    let mut buffer = String::new();
    let _ = stdin().read_to_string(&mut buffer);
    let conf = Ini::from_buffer(buffer);

    let plot = plot_from_config(&conf);
    let fields = fields_from_config(&conf);

    let optical_energy: f64 = conf.get("phonons", "optical_energy").unwrap_or(5e-2);
    let optical_constant: f64 = conf.get("phonons", "optical_constant").unwrap_or(1.5e-3);
    let acoustic_constant: f64 = conf.get("phonons", "acoustic_constant").unwrap_or(1.5e-3);
    let m = SL3::new(optical_energy, optical_constant, acoustic_constant);

    let dt: f64 = conf.get("modelling", "dt").unwrap_or(1e-1);
    let all_time: f64 = conf.get("modelling", "all_time").unwrap_or(1e3);
    let temperature: f64 = conf.get("modelling", "temperature").unwrap_or(7e-3);
    let particles: usize = conf.get("modelling", "particles").unwrap_or(100);
    let threads: usize = conf.get("modelling", "threads").unwrap_or(1);

    let plot_count = ((plot.high - plot.low) / plot.step) as u32;
    let all_time_start = SteadyTime::now();

    println!("{:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} \
                  {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} {:^10} \
                  {:^10} {:^10} {:^10}",
             "E0.x",
             "E0.y",
             "E1.x",
             "E1.y",
             "E2.x",
             "E2.y",
             "B0",
             "B1",
             "B2",
             "omega1",
             "omega2",
             "phi",
             "jx",
             "jx_std",
             "jy",
             "jy_std",
             "optical",
             "acoustic",
             "tau");
    for (index, f) in plot.domain(&fields).enumerate() {
        let part_time_start = SteadyTime::now();
        let ensemble = create_ensemble(particles, &m, temperature);

        let mut ensemble_summary = vec![Summary::empty(); particles];
        let mut pool = Pool::new(threads as u32);

        pool.scoped(|scope| {
            for (index, item) in ensemble_summary.iter_mut().enumerate() {
                let dt = dt;
                let all_time = all_time;
                let ref fields = f;
                let ref particle = ensemble[index];
                scope.execute(move || {
                    *item = particle.run(dt, all_time, fields);
                });
            }
        });

        let mut result = Stats::from_ensemble(&ensemble_summary);
        // dirty: electrons have negative charge
        result.current = -result.current;
        println!("{:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} \
                  {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} \
                  {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e} {:^10.3e}",
                 f.e.0.x,
                 f.e.0.y,
                 f.e.1.x,
                 f.e.1.y,
                 f.e.2.x,
                 f.e.2.y,
                 f.b.0,
                 f.b.1,
                 f.b.2,
                 f.omega.1,
                 f.omega.2,
                 f.phi,
                 result.current.x,
                 result.current_std.x,
                 result.current.y,
                 result.current_std.y,
                 result.optical,
                 result.acoustic,
                 result.tau);
    }
}



struct Plot {
    low: f64,
    high: f64,
    step: f64,
    var: String,
    fields: Fields,
    n: usize,
    current: usize,
}

impl Iterator for Plot {
    type Item = Fields;
    fn next(&mut self) -> Option<Fields> {
        if self.n == 0 {
            self.n = ((self.high - self.low) / self.step) as usize;
        }
        let mut fields = self.fields.clone();
        let value = self.low + self.step * self.current as f64;
        match self.var.as_ref() {
            "E0.x" => fields.e.0.x = value,
            "E0.y" => fields.e.0.y = value,
            "E1.x" => fields.e.1.x = value,
            "E1.y" => fields.e.1.y = value,
            "E2.x" => fields.e.2.x = value,
            "E2.y" => fields.e.2.y = value,
            "B0" => fields.b.0 = value,
            "B1" => fields.b.1 = value,
            "B2" => fields.b.2 = value,
            "phi" => fields.phi = value,
            _ => {
                println!("something went wrong");
                return None;
            }
        }
        if self.current < self.n {
            self.current += 1;
            Some(fields)
        } else {
            None
        }
    }
}


impl Plot {
    pub fn domain(mut self, f: &Fields) -> Self {
        self.fields = f.clone();
        self
    }
}

fn plot_from_config(conf: &Ini) -> Plot {
    let low: f64 = conf.get("plot", "low").unwrap_or(0.0);
    let high: f64 = conf.get("plot", "high").unwrap_or(0.0);
    let step: f64 = conf.get("plot", "step").unwrap_or(1.0);
    let var: String = conf.get("plot", "var").unwrap_or("E0.y".to_owned());
    Plot {
        low: low,
        high: high,
        step: step,
        var: var,
        fields: Fields::zero(),
        n: 0,
        current: 0,
    }
}

fn fields_from_config(conf: &Ini) -> Fields {
    let mut f = Fields::zero();
    f.e.0 = conf.get("fields", "E0").unwrap_or(f.e.0);
    f.e.1 = conf.get("fields", "E1").unwrap_or(f.e.1);
    f.e.2 = conf.get("fields", "E2").unwrap_or(f.e.2);
    f.b.0 = conf.get("fields", "B0").unwrap_or(f.b.0);
    f.b.1 = conf.get("fields", "B1").unwrap_or(f.b.1);
    f.b.2 = conf.get("fields", "B2").unwrap_or(f.b.2);
    f.omega.1 = conf.get("fields", "omega1").unwrap_or(f.omega.1);
    f.omega.2 = conf.get("fields", "omega2").unwrap_or(f.omega.2);
    f.phi = conf.get("fields", "phi").unwrap_or(f.phi);
    f
}
