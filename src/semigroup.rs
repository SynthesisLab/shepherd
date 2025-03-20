use crate::flow;
use crate::ideal;
use crate::nfa;
use log::debug;
use std::collections::HashSet; // for distinct method
use std::collections::VecDeque;
use std::fmt;

pub struct FlowSemigroup {
    flows: HashSet<flow::Flow>,
}

impl FlowSemigroup {
    pub fn new() -> Self {
        FlowSemigroup {
            flows: HashSet::new(),
        }
    }

    pub fn compute(flows: &HashSet<flow::Flow>) -> Self {
        let mut semigroup = FlowSemigroup::new();
        for flow in flows.iter() {
            semigroup.flows.insert(flow.clone());
        }
        semigroup.close_by_product_and_iteration();
        semigroup
    }

    #[allow(dead_code)]
    pub fn contains(&self, flow: &flow::Flow) -> bool {
        Self::is_covered(flow, &self.flows)
    }

    pub fn get_path_problem_solution(&self, target: &[nfa::State]) -> ideal::Ideal {
        ideal::Ideal::from_vec(
            &self
                .flows
                .iter()
                .map(|flow| flow.pre_image(target))
                .collect::<Vec<_>>(),
        )
    }

    fn close_by_product_and_iteration(&mut self) {
        let mut to_process: VecDeque<flow::Flow> = self.flows.iter().cloned().collect();
        let mut processed = HashSet::<flow::Flow>::new();
        while !to_process.is_empty() {
            let flow = to_process.pop_front().unwrap();
            debug!(
                "\nClose_by_product_and_iteration processing flow\n{}\n",
                flow
            );
            if Self::is_covered(&flow, &processed) {
                //debug!("Skipped inqueue\n{}", flow);
                continue;
            }
            processed.insert(flow.clone());

            let iteration = flow.iteration();
            if !Self::is_covered(&iteration, &self.flows) {
                print!("\n\nAdded iteration\n{}", iteration);
                self.flows.insert(iteration.clone());
                to_process.push_back(iteration);
            } else {
                //debug!("\n\nSkipped iteration\n{}", iteration);
            }
            {
                let right_products = self.flows.iter().map(|other| &flow * other);
                let left_products = self.flows.iter().map(|other| other * &flow);
                let products: HashSet<flow::Flow> = left_products.chain(right_products).collect();
                for product in products {
                    if !Self::is_covered(&product, &self.flows) {
                        print!("\n\nAdded product\n{}", product);
                        self.flows.insert(product.clone());
                        to_process.push_back(product);
                    } else {
                        //debug!("\n\nSkipped product\n{}", product);
                    }
                }
            }
            self.minimize();
        }
    }

    fn is_covered(flow: &flow::Flow, others: &HashSet<flow::Flow>) -> bool {
        /*debug!(
            "Checking whether\n{} is covered by\n{}\n",
            flow,
            others
                .iter()
                .map(flow::Flow::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        );*/
        return others.iter().any(|other| flow <= other);
    }

    fn minimize(&mut self) {
        print!("Minimizing semigroup");
        let before = self.flows.len();
        //debug!("Before minimization\n{}", self);
        let mut to_remove = HashSet::new();
        for flow in self.flows.iter() {
            if to_remove.contains(flow) {
                continue;
            }
            let upper_bound = self.flows.iter().filter(|other| flow < other).next();
            match upper_bound {
                Some(upper_bound) => {
                    print!("Removing\n{}\nwhich is covered by\n{}\n", flow, upper_bound);
                    to_remove.insert(flow.clone());
                }
                None => {}
            }
        }
        for flow in to_remove.iter() {
            self.flows.remove(flow);
        }
        //debug!("After minimization\n{}", self);
        let after = self.flows.len();
        debug!(
            "Minimized semigroup from {} flows to {} flows",
            before, after
        );
    }
}

impl fmt::Display for FlowSemigroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut flows = self
            .flows
            .iter()
            .map(|flow| flow.to_string())
            .collect::<Vec<_>>();
        flows.sort();
        write!(f, "{}", flows.join("\r\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef::{C0, C1, OMEGA};
    use crate::flow::Flow;

    #[test]
    fn test_flow_semigroup_compute1() {
        let flowa = Flow::from_lines(&[&[OMEGA, C1], &[C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa].into();
        let semigroup = FlowSemigroup::compute(&flows);
        let flow_omega = Flow::from_entries(2, &[OMEGA, OMEGA, C0, OMEGA]);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.flows.contains(&flow_omega));
    }

    #[test]
    fn test_flow_semigroup_compute2() {
        let flowa = Flow::from_lines(&[&[OMEGA, OMEGA, C0], &[OMEGA, OMEGA, C1], &[C0, C0, OMEGA]]);
        let flowb = Flow::from_lines(&[&[OMEGA, C0, C0], &[C0, C1, C0], &[C0, C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa.clone(), flowb.clone()].into();
        let semigroup = FlowSemigroup::compute(&flows);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.contains(&flowa));
        assert!(semigroup.contains(&flowb));
    }

    #[test]
    fn test_flow_semigroup_compute3() {
        let flowa = Flow::from_lines(&[&[OMEGA, C1, C0], &[OMEGA, C0, C1], &[C0, C0, OMEGA]]);
        let flowb = Flow::from_lines(&[&[OMEGA, C0, C0], &[C0, C1, C0], &[C0, C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa.clone(), flowb.clone()].into();
        let semigroup = FlowSemigroup::compute(&flows);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.contains(&flowa));
        assert!(semigroup.contains(&flowb));
    }
}
