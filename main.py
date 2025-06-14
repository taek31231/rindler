import streamlit as st
import plotly.graph_objects as go
import numpy as np

# 상수 설정
C = 1.0  # 광속을 1로 설정하여 c*t 대신 t 사용 (단위 조절)

def create_rindler_diagram(acceleration):
    """
    린들러 시공간 도표를 생성합니다.
    Args:
        acceleration (float): 린들러 관찰자의 고유 가속도 (a).
    """
    fig = go.Figure()

    # --- 1. 민코프스키 배경 설정 ---
    axis_range = 5  # 축 범위
    num_points = 200 # 곡선 그리기 위한 점 개수

    # x-axis (공간)
    fig.add_shape(type="line", x0=-axis_range, y0=0, x1=axis_range, y1=0,
                  line=dict(color="blue", width=2), name="Minkowski x-axis")
    # t-axis (시간)
    fig.add_shape(type="line", x0=0, y0=-axis_range, x1=0, y1=axis_range,
                  line=dict(color="blue", width=2), name="Minkowski t-axis")

    # --- 2. 린들러 지평선 (Rindler Horizon) ---
    # 이는 민코프스키 시공간의 빛 원뿔과 일치합니다.
    # 미래 지평선 (x=t)
    fig.add_shape(type="line", x0=-axis_range, y0=-axis_range, x1=axis_range, y1=axis_range,
                  line=dict(color="gray", width=2, dash="dashdot"), name="Rindler Horizon (x=t)")
    # 과거 지평선 (x=-t)
    fig.add_shape(type="line", x0=-axis_range, y0=axis_range, x1=axis_range, y1=-axis_range,
                  line=dict(color="gray", width=2, dash="dashdot"), name="Rindler Horizon (x=-t)")

    # 지평선 라벨
    fig.add_annotation(x=axis_range-0.5, y=axis_range-0.5, text="Future Rindler Horizon", showarrow=False, font=dict(color="gray"))
    fig.add_annotation(x=axis_range-0.5, y=-(axis_range-0.5), text="Past Rindler Horizon", showarrow=False, font=dict(color="gray"))

    # --- 3. 린들러 관찰자의 세계선 (가속하는 세계선) ---
    # x^2 - t^2 = constant (쌍곡선)
    # Rindler observer worldlines are hyperbolae x^2 - t^2 = (1/a)^2
    # For a given a, different hyperbolae represent observers with different 'distances' from the horizon.
    
    # 1/a 값을 스케일링하여 적절한 범위의 쌍곡선을 그립니다.
    # a가 작을수록 쌍곡선이 더 "평탄"해집니다.
    # a가 클수록 쌍곡선이 더 "날카롭게" 꺾입니다.
    
    # Rindler Observer at xi = 1/a
    if acceleration > 0:
        xi_val_for_observer = 1.0 / acceleration # 표준 린들러 관찰자 세계선
    else:
        xi_val_for_observer = 1.0 # a가 0일 경우 (민코프스키 평행선처럼 보임)

    tau_values = np.linspace(-3, 3, num_points) # 고유 시간 범위

    # 린들러 변환 (Minkowski (t, x) to Rindler (tau, xi))
    # t = xi * sinh(a*tau)
    # x = xi * cosh(a*tau)

    # 린들러 관찰자 세계선 (xi = constant, tau varies)
    # 특정 xi_constant 값에 대한 세계선
    xi_constants = [0.5, 1.0, 2.0] # 여러 쌍곡선 그리기
    
    # x-t 평면에서 린들러 관찰자의 세계선 (쌍곡선)
    for i, xi_c in enumerate(xi_constants):
        if acceleration > 0:
            # 린들러 좌표를 민코프스키 좌표로 변환
            t_rindler_observer = xi_c * np.sinh(acceleration * tau_values)
            x_rindler_observer = xi_c * np.cosh(acceleration * tau_values)

            # 미래 방향 (t > 0)
            valid_indices_future = (x_rindler_observer > 0) & (t_rindler_observer >= -axis_range) & (t_rindler_observer <= axis_range) & \
                                   (x_rindler_observer >= -axis_range) & (x_rindler_observer <= axis_range)
            fig.add_trace(go.Scatter(x=x_rindler_observer[valid_indices_future], y=t_rindler_observer[valid_indices_future], mode='lines',
                                     line=dict(color=f'rgba(255,165,0,{0.5 + 0.1*i})', width=2), name=f'Rindler Worldline (ξ={xi_c:.1f})'))
            # 과거 방향 (t < 0) - 대칭적으로 그립니다.
            fig.add_trace(go.Scatter(x=x_rindler_observer[valid_indices_future], y=-t_rindler_observer[valid_indices_future], mode='lines',
                                     line=dict(color=f'rgba(255,165,0,{0.5 + 0.1*i})', width=2), showlegend=False)) # showlegend=False로 중복 표시 방지

        else: # a = 0 (민코프스키 공간, 정지 또는 등속)
            # a=0 이면 Rindler transform은 t=xi*0, x=xi*1 이 되므로
            # t = 0 (x축), x = xi (일정한 공간값)
            # 이는 린들러 공간이라기 보다는 민코프스키 공간의 일반적인 세계선이 됩니다.
            # 이 경우 xi_c는 단순히 x축 상의 위치가 됩니다.
            fig.add_trace(go.Scatter(x=np.full(num_points, xi_c), y=np.linspace(-axis_range, axis_range, num_points), mode='lines',
                                     line=dict(color='orange', width=2, dash='dot'), name=f'Observer (x={xi_c})'))


    # --- 4. 린들러 좌표 그리드 ---
    # 등-고유시간 선 (Constant tau lines): t = tan(a*tau) * x
    # t/x = tanh(a*tau) 이므로, tanh(a*tau)는 기울기가 됨.
    # a*tau 값이 작으면 기울기가 작고, a*tau 값이 커지면 기울기가 1에 가까워짐.
    tau_grid_values = np.linspace(-2.5, 2.5, 6) # tau 값 범위 설정 (간격 조절)
    for tau_g in tau_grid_values:
        if acceleration > 0:
            slope = np.tanh(acceleration * tau_g)
            x_line = np.linspace(-axis_range, axis_range, num_points)
            t_line = slope * x_line
            
            # 지평선을 넘어가지 않도록 유효 범위만 플로팅
            valid_indices = (np.abs(x_line) < np.abs(t_line) if np.abs(slope) > 1 else np.abs(t_line) < np.abs(x_line)) # 빛원뿔 안쪽 또는 바깥쪽
            valid_indices = (np.abs(t_line) <= axis_range) & (np.abs(x_line) <= axis_range)
            
            fig.add_trace(go.Scatter(x=x_line[valid_indices], y=t_line[valid_indices], mode='lines',
                                     line=dict(color='green', width=1, dash='dot'), name=f'Constant τ={tau_g:.1f}', showlegend=False))

    # 등-고유공간 선 (Constant xi lines): x^2 - t^2 = xi^2 (쌍곡선)
    # 이미 린들러 세계선으로 그렸으므로, 추가하지 않아도 됨.
    # xi_constants = [0.5, 1.0, 2.0] # 위에 사용된 값과 동일


    # --- 5. 레이아웃 설정 ---
    fig.update_layout(
        title=f'린들러 시공간 도표 (고유 가속도 a = {acceleration:.2f})',
        xaxis_title='x (공간)',
        yaxis_title='t (시간)',
        xaxis_range=[-axis_range, axis_range],
        yaxis_range=[-axis_range, axis_range],
        width=700,
        height=700,
        hovermode="closest",
        showlegend=True,
        # Aspect ratio 1:1 유지
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

    return fig

# --- Streamlit 앱 인터페이스 ---
st.set_page_config(layout="wide")

st.title("린들러 시공간 도표")

st.write("""
이 앱은 민코프스키 시공간 내에서 **균일하게 가속하는 관찰자**의 시공간 (린들러 공간)을 시각화합니다.
슬라이더를 조작하여 린들러 관찰자의 **고유 가속도(a)**를 변경해 보세요.
""")

# 사이드바에서 사용자 입력 받기
st.sidebar.header("설정")
# 가속도는 0보다 커야 쌍곡선 세계선이 명확히 보임.
# 너무 큰 값은 도표 범위를 벗어나므로 적절한 Max 값 설정
rindler_acceleration = st.sidebar.slider("고유 가속도 a", min_value=0.01, max_value=2.0, value=0.5, step=0.01)

# 도표 생성 및 표시
rindler_fig = create_rindler_diagram(rindler_acceleration)
st.plotly_chart(rindler_fig, use_container_width=True)

st.subheader("설명")
st.markdown(r"""
* **파란색 축**: 중력이 없는 평탄한 민코프스키 시공간의 시간(t) 축과 공간(x) 축입니다.
* **회색 점선 (린들러 지평선)**: $x = \pm t$ 직선으로, 빛의 세계선과 일치합니다. 이는 린들러 관찰자가 아무리 가속해도 도달하거나 관측할 수 없는 시공간 영역과의 경계입니다. 이는 블랙홀의 사건의 지평선과 유사한 개념으로, **가속으로 인해 발생하는 시야 한계**입니다.
* **주황색 선 (린들러 관찰자의 세계선)**: 균일한 고유 가속도 $a={rindler_acceleration:.2f}$로 움직이는 관찰자들의 궤적입니다. 이들은 민코프스키 시공간에서 쌍곡선 형태를 띱니다.
    * $x^2 - t^2 = \text{상수}$ 의 형태를 가지며, 이 상수는 관찰자와 지평선 사이의 "거리"와 관련이 있습니다.
    * $a$ 값이 커질수록 쌍곡선이 $x$축에 더 가깝게 "눕는" 것을 볼 수 있습니다. 이는 동일한 고유 공간 거리 $\xi$를 가진 관찰자가 더 큰 가속도로 움직일수록 $x$축에 더 가깝게 위치하게 됨을 의미합니다.
* **초록색 점선 (등-고유시간 선)**: 린들러 관찰자 계에서 고유 시간 $\tau$가 일정한 점들을 연결한 선입니다. 민코프스키 도표에서는 원점을 지나는 직선으로 나타납니다.
""")

st.markdown("""
---
**린들러 공간과 블랙홀의 사건의 지평선:**
린들러 지평선은 블랙홀의 사건의 지평선과 몇 가지 유사점을 가집니다. 둘 다 어떤 영역에서는 빛조차 탈출할 수 없는 "경계"를 의미합니다. 하지만 중요한 차이점이 있습니다:
* **린들러 지평선**: 평탄한 시공간에서 **가속**으로 인해 발생하는 현상입니다. 이 지평선은 "실제" 시공간의 곡률이 아니며, 관찰자의 가속도에 의해 나타나는 상대적인 개념입니다. 린들러 관찰자는 가속을 멈추면 이 지평선을 넘어설 수 있습니다.
* **블랙홀의 사건의 지평선**: 시공간의 **극심한 곡률(중력)** 때문에 발생하는 물리적인 경계입니다. 이 지평선을 넘어서면 가속을 멈추더라도 다시는 탈출할 수 없습니다.

따라서 린들러 공간은 일반 상대성 이론의 복잡한 블랙홀 시공간을 이해하기 위한 **개념적인 다리 역할**을 한다고 볼 수 있습니다.
""")

# --- 새로운 린들러 고유 좌표계 도표 생성 함수 ---
def create_rindler_proper_coordinate_diagram(acceleration):
    """
    린들러 고유 좌표계 (tau, xi)에서 시공간 도표를 생성합니다.
    여기서 빛의 세계선은 지수 함수 형태를 띱니다.
    Args:
        acceleration (float): 린들러 관찰자의 고유 가속도 (a).
    """
    fig = go.Figure()

    # 축 범위 설정
    # xi는 0보다 커야 하며, 너무 큰 값은 비현실적이므로 적절한 범위 설정
    xi_min, xi_max = 0.05, 5.0 # xi = 0은 지평선이므로 0.05부터 시작
    tau_min, tau_max = -3.0, 3.0 # 고유 시간 tau 범위
    num_points = 200

    # --- 1. 축 설정 (xi, tau) ---
    fig.add_shape(type="line", x0=xi_min, y0=0, x1=xi_max, y1=0,
                  line=dict(color="blue", width=2), name="xi-axis (Proper Space)")
    fig.add_shape(type="line", x0=0, y0=tau_min, x1=0, y1=tau_max,
                  line=dict(color="blue", width=2), name="tau-axis (Proper Time)")

    # --- 2. 린들러 지평선 (Horizon at xi=0) ---
    fig.add_shape(type="line", x0=0, y0=tau_min, x1=0, y1=tau_max,
                  line=dict(color="red", width=3, dash="dash"), name="Rindler Horizon (xi=0)")
    fig.add_annotation(x=0.1, y=tau_max - 0.2, text="Rindler Horizon (ξ=0)",
                       showarrow=False, font=dict(color="red"), xanchor="left")

    # --- 3. 린들러 관찰자의 세계선 (xi = constant) ---
    # 가속하는 관찰자 자신은 자신의 고유 좌표계에서 정지해 있음
    # 예를 들어 xi = 1/acceleration 에 해당하는 관찰자의 세계선을 그립니다.
    if acceleration > 0:
        observer_xi = 1.0 / acceleration # 예시로 1/a 위치의 관찰자를 그림
        if observer_xi > xi_min and observer_xi < xi_max: # 축 범위 내에 있을 때만 그리기
            fig.add_shape(type="line", x0=observer_xi, y0=tau_min, x1=observer_xi, y1=tau_max,
                          line=dict(color="orange", width=2, dash="dot"), name=f"Rindler Observer (ξ={observer_xi:.2f})")
            fig.add_annotation(x=observer_xi, y=tau_max - 0.4, text=f"Observer (ξ={observer_xi:.2f})",
                               showarrow=False, font=dict(color="orange"), xanchor="center")


    # --- 4. 빛의 세계선 (Light Cones) - 지수 함수 형태 ---
    # d(xi)/d(tau) = +/- a*xi
    # -> xi = xi_0 * exp(a*tau) 또는 xi = xi_0 * exp(-a*tau)
    
    tau_values = np.linspace(tau_min, tau_max, num_points)

    # 여러 초기 xi_0 값에서 출발하는 빛의 경로를 그립니다.
    # 지평선 (xi=0)에 가까운 곳에서 시작하는 빛을 보여주는 것이 중요
    xi0_starts = np.linspace(0.1, xi_max, 5) # 0.1부터 xi_max까지 5개의 시작점

    for i, xi_start in enumerate(xi0_starts):
        # 긍정적인 방향으로의 빛 (xi 증가)
        xi_light_forward = xi_start * np.exp(acceleration * tau_values)
        # 축 범위 내의 유효한 점들만 플로팅
        valid_indices_fwd = (xi_light_forward >= xi_min) & (xi_light_forward <= xi_max)

        fig.add_trace(go.Scatter(x=xi_light_forward[valid_indices_fwd], y=tau_values[valid_indices_fwd],
                                 mode='lines',
                                 line=dict(color=f'rgba(0,128,0,{0.5 + i*0.1})', width=1.5, dash='solid'),
                                 name=f'Light Ray (fwd, ξ0={xi_start:.1f})', showlegend=False))

        # 부정적인 방향으로의 빛 (xi 감소)
        xi_light_backward = xi_start * np.exp(-acceleration * tau_values)
        # 축 범위 내의 유효한 점들만 플로팅
        valid_indices_bwd = (xi_light_backward >= xi_min) & (xi_light_backward <= xi_max)

        fig.add_trace(go.Scatter(x=xi_light_backward[valid_indices_bwd], y=tau_values[valid_indices_bwd],
                                 mode='lines',
                                 line=dict(color=f'rgba(0,128,0,{0.5 + i*0.1})', width=1.5, dash='solid'),
                                 name=f'Light Ray (bwd, ξ0={xi_start:.1f})', showlegend=False))

    # --- 5. 레이아웃 설정 ---
    fig.update_layout(
        title=f'린들러 고유 좌표계 시공간 도표 (a = {acceleration:.2f})',
        xaxis_title='ξ (공간 좌표)',
        yaxis_title='τ (고유 시간)',
        xaxis_range=[xi_min, xi_max],
        yaxis_range=[tau_min, tau_max],
        width=700,
        height=700,
        hovermode="closest",
        showlegend=True,
        yaxis=dict(scaleanchor="x", scaleratio=1), # 스케일 비율 유지
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

    return fig


# --- Streamlit 앱 인터페이스 (기존 코드에 이어서) ---
# ... (기존 create_rindler_diagram 호출 부분) ...

st.markdown("---") # 구분선 추가
st.subheader("린들러 고유 좌표계 시공간 도표 (가속 관찰자 시점)")

st.write("""
이 도표는 균일하게 가속하는 린들러 관찰자 자신의 고유 좌표계 (ξ, τ)에서 시공간이 어떻게 보이는지 시각화합니다.
""")

# 새로운 도표 생성 및 표시
rindler_proper_fig = create_rindler_proper_coordinate_diagram(rindler_acceleration)
st.plotly_chart(rindler_proper_fig, use_container_width=True)

st.markdown(r"""
* **빨간색 점선 (린들러 지평선)**: $\xi = 0$에 위치합니다. 린들러 관찰자에게는 이 지평선 너머의 사건들이 영원히 도달할 수 없는 영역입니다.
* **주황색 점선 (린들러 관찰자의 세계선)**: 린들러 관찰자 자신은 자신의 고유 공간 좌표에 대해 정지해 있으므로, 이 도표에서는 $\xi = \text{상수}$인 수직선으로 나타납니다.
* **초록색 실선 (빛의 세계선)**: 이 가속하는 좌표계에서 빛은 $\xi = \xi_0 e^{\pm a\tau}$ 형태의 **지수 함수 곡선**을 따라 이동하는 것처럼 보입니다.
    * 이 곡선들이 $\xi = 0$ (지평선)으로 수렴하는 것을 볼 수 있습니다. 이는 지평선 근처에서는 빛이 린들러 관찰자에게서 멀어지거나(긍정적인 방향) 다가오는(부정적인 방향) 속도가 느려지는 것처럼 보이는 현상을 반영합니다.
* **가속도 $a$가 커질수록**, 빛의 세계선(초록색 곡선)이 $\xi=0$ 지평선에 더 가파르게 수렴하는 것을 확인할 수 있습니다. 이는 강한 가속이 빛의 궤적을 더 극적으로 휘게 만드는 것처럼 보이는 효과를 나타냅니다.
""")

# --- 등가 원리 시각화 함수 ---
def create_equivalence_principle_diagram(acceleration):
    """
    등가 원리를 시각화합니다: 가속하는 엘리베이터와 중력장 내의 빛의 경로 비교.
    Args:
        acceleration (float): 엘리베이터의 가속도 (a) 또는 중력장 세기 (g).
    """
    fig = go.Figure()

    plot_range = 5 # 도표 범위
    num_points = 100

    # --- 1. 가속하는 엘리베이터 시나리오 (빛이 휘는 것처럼 보임) ---
    # 엘리베이터 한쪽 벽에서 수평으로 빛을 쏘고, 엘리베이터가 위로 가속하는 상황
    # 엘리베이터 안의 관찰자에게는 빛이 포물선으로 휘는 것처럼 보임
    # y = -(1/2) * (a/c^2) * x^2 (근사적으로)
    # 여기서는 시간 축을 제거하고, 공간 축 x와 가속 방향 축 y를 사용합니다.

    x_values_accel = np.linspace(0, plot_range, num_points)
    # y = -0.5 * (acceleration / (C**2)) * x^2  (C=1이므로 y = -0.5 * acceleration * x^2)
    # 가속도가 0일 때 직선, 가속도가 클수록 더 많이 휨
    y_values_accel = -0.5 * acceleration * x_values_accel**2

    fig.add_trace(go.Scatter(x=x_values_accel, y=y_values_accel, mode='lines',
                             line=dict(color="orange", width=3),
                             name=f'Light in Accelerating Elevator (a={acceleration:.2f})'))

    # --- 2. 균일한 중력장 시나리오 (실제로 빛이 휨) ---
    # 중력 g 아래에서 수평으로 발사된 물체의 궤적과 유사 (빛은 다름, 하지만 비유)
    # (실제 빛의 중력 렌즈 현상은 더 복잡하지만, 여기서는 개념 설명을 위한 단순화)
    # 등가 원리를 설명하기 위해, 가속 a와 동일한 크기의 중력장 g를 가정합니다.
    
    # 중력장 내에서 빛이 '휘는' 정도를 시각적으로 유사하게 표현
    y_values_gravity = -0.5 * acceleration * x_values_accel**2 # 동일한 수식을 사용하여 시각적 유사성 강조

    fig.add_trace(go.Scatter(x=x_values_accel, y=y_values_gravity, mode='lines',
                             line=dict(color="purple", width=3, dash="dot"),
                             name=f'Light in Gravitational Field (g={acceleration:.2f})')) # g = a 가정

    # --- 3. 레이아웃 설정 ---
    fig.update_layout(
        title=f'등가 원리 시각화 (가속도/중력장 세기 = {acceleration:.2f})',
        xaxis_title='수평 거리',
        yaxis_title='수직 변위',
        xaxis_range=[0, plot_range],
        yaxis_range=[-plot_range, 0.5], # 음수 방향으로 휘는 것을 고려
        width=700,
        height=500,
        hovermode="closest",
        showlegend=True
    )
    fig.update_xaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')

    return fig

# --- Streamlit 앱 인터페이스 (기존 코드에 이어서) ---
# ... (기존 create_rindler_proper_coordinate_diagram 호출 부분) ...

st.markdown("---") # 구분선 추가
st.subheader("등가 원리 시각화: 가속 vs. 중력")

st.write("""
등가 원리(Equivalence Principle)는 **균일하게 가속하는 기준계**와 **균일한 중력장 내의 정지한 기준계**가 물리적으로 구별될 수 없다는 아인슈타인의 중요한 통찰입니다.
아래 도표는 가속하는 엘리베이터 안에서 빛이 휘는 것처럼 보이는 현상과, 중력장 내에서 빛이 휘는 현상을 비교하여 이를 시각적으로 보여줍니다.
(빛의 실제 중력 렌즈 현상은 더 복잡하지만, 여기서는 등가 원리의 개념적 유사성을 강조하기 위한 단순화된 모델입니다.)
""")

# 등가 원리 도표 생성 및 표시
equivalence_fig = create_equivalence_principle_diagram(rindler_acceleration) # 기존 슬라이더 값 활용
st.plotly_chart(equivalence_fig, use_container_width=True)

st.markdown(r"""
* **주황색 실선**: 위로 가속하는 엘리베이터 안에서 수평으로 발사된 빛의 경로입니다. 엘리베이터 내부의 관찰자에게는 빛이 포물선으로 휘는 것처럼 보입니다.
* **보라색 점선**: 균일한 중력장 내에서 수평으로 발사된 빛의 경로입니다. 중력에 의해 빛이 휘어지는 실제 현상을 단순화하여 보여줍니다.
* **관찰**: 두 곡선이 동일한 가속도(또는 중력장 세기) 값에서 **같은 형태**로 휘어지는 것을 확인할 수 있습니다. 이는 가속이 중력과 동일한 효과를 낸다는 등가 원리의 핵심 아이디어입니다.
""")
