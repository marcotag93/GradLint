use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::error::{Error, Result};

pub use rayon::current_num_threads;

pub struct Execution {
    pool: Option<ThreadPool>,
}

impl Execution {
    pub fn new(threads: Option<usize>) -> Result<Self> {
        let pool = match threads {
            Some(0) => return Err(Error::InvalidThreads),
            Some(n) => Some(ThreadPoolBuilder::new().num_threads(n).build()?),
            None => None,
        };
        Ok(Self { pool })
    }

    pub fn run<T: Send>(&self, operation: impl FnOnce() -> T + Send) -> T {
        match &self.pool {
            Some(pool) => pool.install(operation),
            None => operation(),
        }
    }

    pub fn num_threads(&self) -> usize {
        self.run(rayon::current_num_threads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn explicit_pools_are_local_and_do_not_change_the_default() {
        let default = Execution::new(None).unwrap().num_threads();
        for n in [1, 2, 4] {
            let execution = Execution::new(Some(n)).unwrap();
            assert_eq!(execution.num_threads(), n);
            execution.run(|| {
                assert!((0..64)
                    .into_par_iter()
                    .all(|_| rayon::current_num_threads() == n));
                assert_eq!(Execution::new(None).unwrap().num_threads(), n);
            });
            assert_eq!(Execution::new(None).unwrap().num_threads(), default);
        }
    }

    #[test]
    fn zero_threads_is_rejected() {
        assert!(matches!(
            Execution::new(Some(0)),
            Err(Error::InvalidThreads)
        ));
    }
}
