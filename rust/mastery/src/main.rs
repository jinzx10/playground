extern crate rand;
use rand::Rng;

fn attempt(win_prob: f64, goal: u32, range: u32) -> u32 {
    let mut rng = rand::thread_rng();
    let mut num_games = 0;
    let mut hist: Vec<bool> = vec![false; range as usize];
    let mut mastery: i32 = 0;
    let mut idx: u32 = 0;
    let mut success;
    while mastery < (goal as i32) {
        idx = (idx + 1) % range;
        success = (rng.gen::<f64>()) < win_prob;
        mastery += (success as i32) - (hist[idx as usize] as i32);
        hist[idx as usize] = success;
        num_games += 1;
    }

    num_games
}

fn stats(win_prob: f64, goal: u32, range: u32, num_attempts: u32) -> (f64, f64) {
    let mut mean: f64 = attempt(win_prob, goal, range) as f64;
    let mut variance: f64 = 0.0;
    let mut x: u32;

    for k in 2..num_attempts+1 {
        x = attempt(win_prob, goal, range);

        // incremental mean & variance calculation
        variance = (k - 2) as f64 / (k - 1) as f64 * variance + (x as f64 - mean).powi(2) / k as f64;
        mean += (x as f64 - mean) / k as f64;
    }

    (mean, variance.sqrt())
}

fn main() {
    let num_trials: u32 = 1000000;
    let win_prob: f64 = 2.0/3.0;
    let range: u32 = 8;

    let min_goal: u32 = 7;
    let max_goal: u32 = 7;
    let stride: u32 = 1;

    println!("Mastery/{}      Mean      StdDev    WinProb={:.2}%", range, win_prob*100.0);
    for goal in (min_goal..max_goal+1).step_by(stride as usize) {
        let (mean, std_dev) = stats(win_prob, goal, range, num_trials);
        println!("{:6}       {:8.5}    {:8.5}", goal, mean, std_dev);
    }

}
