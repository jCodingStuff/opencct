fn myfun(x: i32) -> i32 {
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_two_and_two() {
        let result = myfun(4);
        assert_eq!(result, 4);
    }

}
