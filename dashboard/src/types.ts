export type Prediction = {
  date: string;
  tournament: string;
  home_team: string;
  away_team: string;
  model_home_expected_goals: number;
  model_away_expected_goals: number;
  context_home_expected_goals: number;
  context_away_expected_goals: number;
  context_home_attack_multiplier: number;
  context_home_defense_multiplier: number;
  context_away_attack_multiplier: number;
  context_away_defense_multiplier: number;
  context_draw_probability_multiplier: number;
  context_applied: boolean;
  context_confidence: string | null;
  context_notes: string | null;
  home_expected_goals: number;
  away_expected_goals: number;
  goal_inflation: number;
  max_total_candidate_goals: number;
  draw_probability_multiplier: number;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  recommended_scoreline: string;
  recommended_expected_points: number;
  draw_alternative_scoreline: string;
  draw_alternative_expected_points: number;
  draw_alternative_is_competitive: boolean;
  home_elo: number;
  away_elo: number;
  elo_diff: number;
};

export type PredictionAudit = {
  date: string;
  tournament: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  model_home_expected_goals: number;
  model_away_expected_goals: number;
  home_expected_goals: number;
  away_expected_goals: number;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  recommended_home_score: number;
  recommended_away_score: number;
  recommended_scoreline: string;
  recommended_expected_points: number;
  expected_effectiveness: number;
  actual_points: number;
  actual_effectiveness: number;
  actual_category: string;
  result_correct: boolean;
  exact_score: boolean;
};

export type GroupSimulation = {
  team: string;
  group: string;
  first_probability: number;
  second_probability: number;
  third_probability: number;
  fourth_probability: number;
  advance_probability: number;
};

export type TournamentSimulation = {
  team: string;
  group: string;
  round_of_32_probability: number;
  round_of_16_probability: number;
  quarterfinal_probability: number;
  semifinal_probability: number;
  final_probability: number;
  champion_probability: number;
  runner_up_probability: number;
  third_place_probability: number;
  podium_expected_points: number;
};

export type TopScorer = {
  player: string;
  team: string;
  expected_goals: number;
  top_scorer_score: number;
  estimated_team_tournament_goals: number;
};

export type QualificationBacktestStrategy = {
  rank: number;
  goal_inflation: number;
  max_total_candidate_goals: number | null;
  draw_probability_multiplier: number;
  matches: number;
  total_points: number;
  average_points: number;
  exact_scores: number;
  exact_score_rate: number;
  result_accuracy_rate: number;
  winner_and_diff: number;
  draws: number;
  winners: number;
  misses: number;
  miss_rate: number;
  average_pick_goals: number;
  high_total_pick_rate: number;
  selected_strategy: boolean;
  training_window: string;
  validation_window: string;
  training_matches: number;
  training_feature_rows: number;
  validation_matches: number;
  home_mae: number;
  away_mae: number;
};

export type QualificationBacktestPick = {
  date: string;
  home_team: string;
  away_team: string;
  home_score: number;
  away_score: number;
  home_expected_goals: number;
  away_expected_goals: number;
  strategy_home_expected_goals: number;
  strategy_away_expected_goals: number;
  optimized_home_score: number;
  optimized_away_score: number;
  optimized_expected_points: number;
  optimized_points: number;
  optimized_category: string;
  strategy_goal_inflation: number;
  strategy_max_total_candidate_goals: number | null;
  strategy_draw_probability_multiplier: number;
};

export type Metadata = {
  generated_at: string;
  source: string;
  group_simulations: number;
  tournament_simulations: number;
  simulation_seed: number;
};

export type DashboardData = {
  predictions: Prediction[];
  predictionAudit: PredictionAudit[];
  groups: GroupSimulation[];
  tournament: TournamentSimulation[];
  topScorers: TopScorer[];
  qualificationBacktestStrategies: QualificationBacktestStrategy[];
  qualificationBacktestPicks: QualificationBacktestPick[];
  metadata: Metadata;
};
