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
st.markdown(f"""
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
