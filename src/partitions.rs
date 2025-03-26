use cached::proc_macro::cached;
use std::{collections::HashSet, vec::Vec};

/* get all partitions of non-negative integers of length len and sum equal to x.
E.g. if len = 3 and x = 4 returns [[4,0,0], [3,1,0], [3,0,1], [2,2,0], [2,1,1], ...., [0,0,4]]
 */
pub fn get_partitions(x: u16, len: usize) -> Vec<Vec<u16>> {
    let mut result: Vec<Vec<u16>> = Vec::new();
    if len > 0 {
        let mut current = vec![0; len];
        current[0] = x;
        get_partitions_rec(0, &mut current, &mut result);
    }
    result
}

fn get_partitions_rec(start_index: usize, current: &mut Vec<u16>, result: &mut Vec<Vec<u16>>) {
    result.push(current.clone());
    if start_index + 1 >= current.len() {
        return;
    }
    while current[start_index] > 0 {
        current[start_index] -= 1;
        current[start_index + 1] = current.iter().skip(start_index + 1).sum::<u16>() + 1;
        (start_index + 2..current.len()).for_each(|i| {
            current[i] = 0;
        });
        get_partitions_rec(start_index + 1, current, result);
    }
}

//takes a vector of vectors of a generic type and computes its cartesian product
//this if one of the vectors is empty,
//returns an iterator over Vec<T> where T is the generic type
pub fn cartesian_product<T: Clone + Eq + std::hash::Hash>(vectors: &[Vec<T>]) -> Vec<Vec<T>> {
    match vectors.len() {
        0 => vec![],
        1 => vectors[0]
            .iter()
            .map(|x| vec![x.clone()])
            .collect::<Vec<_>>(),
        _ => {
            let mut result = HashSet::new();
            for x in &vectors[0] {
                for mut y in cartesian_product(&vectors[1..]) {
                    y.insert(0, x.clone());
                    result.insert(y);
                }
            }
            result.into_iter().collect::<Vec<Vec<_>>>()
        }
    }
}

#[cached]
pub(crate) fn get_transports(c: u16, len: usize) -> Vec<Vec<u16>> {
    assert!(len > 0);
    let mut result: Vec<Vec<u16>> = Vec::new();
    get_transports_rec(c, vec![0; len], 0, &mut result);
    result
}

fn get_transports_rec(c: u16, mut current: Vec<u16>, pointer: usize, result: &mut Vec<Vec<u16>>) {
    //invariant: current is 0 on indices >= pointer
    if c == 0 {
        result.push(current);
    } else if (pointer + 1) < current.len() {
        for c0 in 0..c {
            let mut current = current.clone();
            current[pointer] = c0;
            get_transports_rec(c - c0, current, pointer + 1, result);
        }
        current[pointer] = c;
        result.push(current);
    } else {
        current[pointer] = c;
        result.push(current);
    }
}

#[cfg(test)]
mod test {
    use crate::partitions::{cartesian_product, get_partitions};

    use super::get_transports;

    //test _get_partitions_rec on an example with start_index=0 current= [3,0,0] and result empty
    #[test]
    fn get_partitions_rec_test() {
        let x = 3;
        let expected = vec![
            vec![3, 0, 0],
            vec![2, 1, 0],
            vec![2, 0, 1],
            vec![1, 2, 0],
            vec![1, 1, 1],
            vec![1, 0, 2],
            vec![0, 3, 0],
            vec![0, 2, 1],
            vec![0, 1, 2],
            vec![0, 0, 3],
        ];
        assert_eq!(get_partitions(x, 3), expected);
    }

    #[test]
    fn cartesian_product_test() {
        let empty = Vec::<u16>::new();
        let empty_vec = Vec::<Vec<u16>>::new();
        assert_eq!(
            cartesian_product(&[empty.clone(), empty.clone(), empty.clone()]),
            empty_vec
        );
    }

    #[test]
    fn get_transports_test() {
        let transports = get_transports(5, 3);
        assert_eq!(
            transports,
            vec![
                [0, 0, 5],
                [0, 1, 4],
                [0, 2, 3],
                [0, 3, 2],
                [0, 4, 1],
                [0, 5, 0],
                [1, 0, 4],
                [1, 1, 3],
                [1, 2, 2],
                [1, 3, 1],
                [1, 4, 0],
                [2, 0, 3],
                [2, 1, 2],
                [2, 2, 1],
                [2, 3, 0],
                [3, 0, 2],
                [3, 1, 1],
                [3, 2, 0],
                [4, 0, 1],
                [4, 1, 0],
                [5, 0, 0]
            ]
        );
    }
}
