export type Prediction = {
  date: string;
  tournament: string;
  home_team: string;
  away_team: string;
  home_expected_goals: number;
  away_expected_goals: number;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  recommended_scoreline: string;
  recommended_expected_points: number;
  home_elo: number;
  away_elo: number;
  elo_diff: number;
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

export type Metadata = {
  generated_at: string;
  source: string;
  tournament_simulations: number;
};

export type DashboardData = {
  predictions: Prediction[];
  groups: GroupSimulation[];
  tournament: TournamentSimulation[];
  topScorers: TopScorer[];
  metadata: Metadata;
};
