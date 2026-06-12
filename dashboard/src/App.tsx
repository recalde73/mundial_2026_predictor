import { useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import type {
  DashboardData,
  GroupSimulation,
  Metadata,
  Prediction,
  TopScorer,
  TournamentSimulation,
} from './types';

type Tab = 'summary' | 'matches' | 'groups' | 'tournament' | 'scorers';

const tabs: Array<{ id: Tab; label: string }> = [
  { id: 'summary', label: 'Resumen' },
  { id: 'matches', label: 'Partidos' },
  { id: 'groups', label: 'Grupos' },
  { id: 'tournament', label: 'Torneo' },
  { id: 'scorers', label: 'Goleador' },
];

const json = async <T,>(path: string): Promise<T> => {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`No se pudo cargar ${path}`);
  }
  return response.json() as Promise<T>;
};

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const num = (value: number, digits = 2) => value.toFixed(digits);

function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('summary');
  const [query, setQuery] = useState('');
  const [group, setGroup] = useState('all');

  useEffect(() => {
    Promise.all([
      json<Prediction[]>('/data/predictions.json'),
      json<GroupSimulation[]>('/data/groups.json'),
      json<TournamentSimulation[]>('/data/tournament.json'),
      json<TopScorer[]>('/data/top_scorers.json'),
      json<Metadata>('/data/metadata.json'),
    ])
      .then(([predictions, groups, tournament, topScorers, metadata]) => {
        setData({ predictions, groups, tournament, topScorers, metadata });
      })
      .catch((caught: unknown) => {
        setError(caught instanceof Error ? caught.message : 'Error desconocido');
      });
  }, []);

  const teams = useMemo(() => {
    if (!data) return [];
    return Array.from(new Set(data.tournament.map((row) => row.team))).sort();
  }, [data]);

  const groups = useMemo(() => {
    if (!data) return [];
    return Array.from(new Set(data.groups.map((row) => row.group))).sort();
  }, [data]);

  if (error) {
    return <Message title="No se pudo cargar el dashboard" detail={error} />;
  }

  if (!data) {
    return <Message title="Cargando dashboard" detail="Leyendo JSON exportados desde el modelo." />;
  }

  const normalizedQuery = query.trim().toLowerCase();
  const teamMatches = (team: string) => !normalizedQuery || team.toLowerCase().includes(normalizedQuery);
  const groupMatches = (rowGroup: string) => group === 'all' || rowGroup === group;

  const filteredPredictions = data.predictions.filter(
    (row) =>
      (teamMatches(row.home_team) || teamMatches(row.away_team)) &&
      (group === 'all' || groupForTeam(data.groups, row.home_team) === group),
  );
  const filteredGroups = data.groups.filter((row) => teamMatches(row.team) && groupMatches(row.group));
  const filteredTournament = data.tournament.filter((row) => teamMatches(row.team) && groupMatches(row.group));
  const filteredScorers = data.topScorers.filter((row) => teamMatches(row.team) || teamMatches(row.player));

  return (
    <main className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Mundial 2026 Predictor</p>
          <h1>Motor de picks, simulaciones y podio</h1>
          <p className="heroText">
            Dashboard generado desde el modelo con datos reales, Elo historico, 100.000 simulaciones y optimizacion por puntos del juego.
          </p>
        </div>
        <div className="heroCard">
          <span>Simulaciones torneo</span>
          <strong>{data.metadata.tournament_simulations.toLocaleString('es-AR')}</strong>
          <small>Actualizado: {new Date(data.metadata.generated_at).toLocaleString('es-AR')}</small>
        </div>
      </header>

      <section className="controls">
        <div className="tabs" role="tablist">
          {tabs.map((item) => (
            <button key={item.id} className={tab === item.id ? 'active' : ''} onClick={() => setTab(item.id)}>
              {item.label}
            </button>
          ))}
        </div>
        <div className="filters">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Buscar equipo o jugador"
            list="teams"
          />
          <datalist id="teams">
            {teams.map((team) => (
              <option key={team} value={team} />
            ))}
          </datalist>
          <select value={group} onChange={(event) => setGroup(event.target.value)}>
            <option value="all">Todos los grupos</option>
            {groups.map((groupName) => (
              <option key={groupName} value={groupName}>{groupName}</option>
            ))}
          </select>
        </div>
      </section>

      {tab === 'summary' && <Summary data={data} />}
      {tab === 'matches' && <MatchesTable rows={filteredPredictions} />}
      {tab === 'groups' && <GroupsTable rows={filteredGroups} />}
      {tab === 'tournament' && <TournamentTable rows={filteredTournament} />}
      {tab === 'scorers' && <ScorersTable rows={filteredScorers} />}
    </main>
  );
}

function Message({ title, detail }: { title: string; detail: string }) {
  return (
    <main className="shell message">
      <h1>{title}</h1>
      <p>{detail}</p>
    </main>
  );
}

function Summary({ data }: { data: DashboardData }) {
  const champion = data.tournament[0];
  const runnerUp = distinctBest(data.tournament, [champion.team], 'runner_up_probability');
  const third = distinctBest(data.tournament, [champion.team, runnerUp.team], 'third_place_probability');
  const scorer = data.topScorers[0];
  const topChampion = data.tournament.slice(0, 8);
  const topMatches = [...data.predictions]
    .sort((a, b) => b.recommended_expected_points - a.recommended_expected_points)
    .slice(0, 8);

  return (
    <section className="grid">
      <article className="panel span2">
        <h2>Podio recomendado</h2>
        <div className="podium">
          <PodiumCard label="Campeon" name={champion.team} value={pct(champion.champion_probability)} />
          <PodiumCard label="Subcampeon" name={runnerUp.team} value={pct(runnerUp.runner_up_probability)} />
          <PodiumCard label="Tercer puesto" name={third.team} value={pct(third.third_place_probability)} />
        </div>
      </article>

      <article className="panel scorerCard">
        <h2>Goleador</h2>
        <strong>{scorer.player}</strong>
        <span>{scorer.team}</span>
        <p>{num(scorer.expected_goals)} goles esperados estimados</p>
      </article>

      <article className="panel">
        <h2>Candidatos al titulo</h2>
        <BarList rows={topChampion.map((row) => ({ label: row.team, value: row.champion_probability }))} />
      </article>

      <article className="panel">
        <h2>Picks mas fuertes</h2>
        <div className="miniList">
          {topMatches.map((row) => (
            <div key={`${row.date}-${row.home_team}-${row.away_team}`}>
              <span>{row.home_team} vs {row.away_team}</span>
              <strong>{row.recommended_scoreline} · {num(row.recommended_expected_points)} pts</strong>
            </div>
          ))}
        </div>
      </article>
    </section>
  );
}

function PodiumCard({ label, name, value }: { label: string; name: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{name}</strong>
      <small>{value}</small>
    </div>
  );
}

function MatchesTable({ rows }: { rows: Prediction[] }) {
  return (
    <TablePanel title="Predicciones partido a partido" count={rows.length}>
      <table>
        <thead>
          <tr>
            <th>Fecha</th>
            <th>Partido</th>
            <th>xG</th>
            <th>1X2</th>
            <th>Pick</th>
            <th>Pts esp.</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.date}-${row.home_team}-${row.away_team}`}>
              <td>{row.date}</td>
              <td><strong>{row.home_team}</strong> vs <strong>{row.away_team}</strong></td>
              <td>{num(row.home_expected_goals)} - {num(row.away_expected_goals)}</td>
              <td>{pct(row.home_win_probability)} / {pct(row.draw_probability)} / {pct(row.away_win_probability)}</td>
              <td><span className="pill">{row.recommended_scoreline}</span></td>
              <td>{num(row.recommended_expected_points)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TablePanel>
  );
}

function GroupsTable({ rows }: { rows: GroupSimulation[] }) {
  return (
    <TablePanel title="Probabilidades de grupo" count={rows.length}>
      <table>
        <thead>
          <tr>
            <th>Grupo</th>
            <th>Equipo</th>
            <th>1ro</th>
            <th>2do</th>
            <th>3ro</th>
            <th>Avanza</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.team}>
              <td>{row.group}</td>
              <td><strong>{row.team}</strong></td>
              <td>{pct(row.first_probability)}</td>
              <td>{pct(row.second_probability)}</td>
              <td>{pct(row.third_probability)}</td>
              <td><ProbabilityBar value={row.advance_probability} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </TablePanel>
  );
}

function TournamentTable({ rows }: { rows: TournamentSimulation[] }) {
  return (
    <TablePanel title="Simulacion completa del torneo" count={rows.length}>
      <table>
        <thead>
          <tr>
            <th>Equipo</th>
            <th>Grupo</th>
            <th>Semis</th>
            <th>Final</th>
            <th>Campeon</th>
            <th>Subcampeon</th>
            <th>3ro</th>
            <th>EV podio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.team}>
              <td><strong>{row.team}</strong></td>
              <td>{row.group}</td>
              <td>{pct(row.semifinal_probability)}</td>
              <td>{pct(row.final_probability)}</td>
              <td><ProbabilityBar value={row.champion_probability} /></td>
              <td>{pct(row.runner_up_probability)}</td>
              <td>{pct(row.third_place_probability)}</td>
              <td>{num(row.podium_expected_points)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TablePanel>
  );
}

function ScorersTable({ rows }: { rows: TopScorer[] }) {
  return (
    <TablePanel title="Goleador del torneo" count={rows.length}>
      <table>
        <thead>
          <tr>
            <th>Jugador</th>
            <th>Equipo</th>
            <th>Goles equipo</th>
            <th>Goles esperados jugador</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.player}-${row.team}`}>
              <td><strong>{row.player}</strong></td>
              <td>{row.team}</td>
              <td>{num(row.estimated_team_tournament_goals)}</td>
              <td><ProbabilityBar value={row.expected_goals / Math.max(rows[0]?.expected_goals ?? 1, 1)} label={num(row.expected_goals)} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </TablePanel>
  );
}

function TablePanel({ title, count, children }: { title: string; count: number; children: ReactNode }) {
  return (
    <section className="panel tablePanel">
      <div className="panelHeader">
        <h2>{title}</h2>
        <span>{count} filas</span>
      </div>
      <div className="tableWrap">{children}</div>
    </section>
  );
}

function ProbabilityBar({ value, label }: { value: number; label?: string }) {
  return (
    <div className="probability">
      <div><span style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} /></div>
      <strong>{label ?? pct(value)}</strong>
    </div>
  );
}

function BarList({ rows }: { rows: Array<{ label: string; value: number }> }) {
  return (
    <div className="barList">
      {rows.map((row) => (
        <div key={row.label}>
          <span>{row.label}</span>
          <ProbabilityBar value={row.value} />
        </div>
      ))}
    </div>
  );
}

function groupForTeam(rows: GroupSimulation[], team: string) {
  return rows.find((row) => row.team === team)?.group;
}

function distinctBest<T extends { team: string } & Record<string, string | number>>(rows: T[], excluded: string[], key: keyof T): T {
  return [...rows]
    .filter((row) => !excluded.includes(String(row.team)))
    .sort((a, b) => Number(b[key]) - Number(a[key]))[0];
}

export default App;
