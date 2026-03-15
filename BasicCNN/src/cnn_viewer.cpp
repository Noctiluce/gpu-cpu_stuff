#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstdio>

#include "tensor.h"
#include "layers.h"

constexpr int WIN_W   = 1920;
constexpr int WIN_H   = 1080;
constexpr int PANEL_W = 900;            // panneau gauche
constexpr int DRAW_W  = WIN_W - PANEL_W;
constexpr int GRID    = 28;
constexpr int CELL    = WIN_H / GRID;   // 38 px par cellule

const sf::Color BG_DARK  { 245, 244, 240 };
const sf::Color BG_PANEL { 251, 250, 246 };
const sf::Color COL_SEP  {  21,  97, 180 };
const sf::Color COL_CYAN {  21,  97, 180 };
const sf::Color COL_WHITE{  15,  15,  20 };
const sf::Color COL_DIM  { 120, 118, 130 };
const sf::Color COL_RED  { 196,  30,  15 };
const sf::Color COL_GREEN{   8, 120,  80 };
const sf::Color COL_GREY { 205, 200, 192 };
const sf::Color COL_BAR  {   8, 120,  80 };
const sf::Color COL_BARG { 210, 130,  30 };

struct Step {
    std::string label;
    sf::Color   label_col;
    int         filters = 0, H = 0, W = 0;
    std::vector<float> data;
    bool               is_bars = false;
    std::vector<float> bars;
    std::string        bars_title;
};

struct CNNResult {
    int   prediction = -1;
    float scores[10] = {};
    std::vector<Step> steps;
};

static Sequential build_cnn() {
    Sequential net;
    net.add<Conv2D>  (1,   8,  3, 1)
       .add<ReLU>    ()
       .add<MaxPool2D>(2)
       .add<Conv2D>  (8,  16,  3, 2)
       .add<ReLU>    ()
       .add<MaxPool2D>(2)
       .add<Flatten> ()
       .add<Linear>  (400, 128, 3)
       .add<ReLU>    ()
       .add<Linear>  (128,  10, 4);
    return net;
}

static CNNResult run_cnn(Sequential& net, const float* px) {
    Tensor input({1, 28, 28});
    std::copy(px, px + 28*28, input.data.begin());

    std::vector<Tensor> acts;
    Tensor logits = net.forwardInspect(input, acts);
    Softmax sm;
    Tensor probs = sm.forward(logits);

    CNNResult res;
    for (int i = 0; i < 10; ++i) res.scores[i] = probs.data[i];
    res.prediction = (int)(std::max_element(probs.data.begin(), probs.data.end())
                           - probs.data.begin());

    auto push3d = [&](int idx, const std::string& lbl, sf::Color col) {
        const Tensor& t = acts[idx];
        Step s;
        s.label = lbl; s.label_col = col;
        s.filters = t.shape[0]; s.H = t.shape[1]; s.W = t.shape[2];
        s.data = t.data;
        res.steps.push_back(std::move(s));
    };
    auto push1d = [&](int idx, const std::string& lbl, sf::Color col,
                      const std::string& bt) {
        const Tensor& t = acts[idx];
        Step s;
        s.label = lbl; s.label_col = col;
        s.is_bars = true; s.bars = t.data; s.bars_title = bt;
        res.steps.push_back(std::move(s));
    };

    push3d(0, "Conv1          [8 x 26x26]",  sf::Color(196,  30,  15));
    push3d(1, "ReLU1          [8 x 26x26]",  sf::Color(210, 110,   0));
    push3d(2, "MaxPool1       [8 x 13x13]",  sf::Color( 21,  97, 180));
    push3d(3, "Conv2          [16 x 11x11]", sf::Color(196,  30,  15));
    push3d(4, "ReLU2          [16 x 11x11]", sf::Color(210, 110,   0));
    push3d(5, "MaxPool2       [16 x 5x5]",   sf::Color( 21,  97, 180));
    push1d(7, "Linear1 + ReLU [128]",        sf::Color(130,  30, 180), "128 neurons");
    push1d(9, "Linear2 / logits [10]",       sf::Color( 15,  15,  15), "logits");

    Step sm_step;
    sm_step.label      = "Softmax        [10]";
    sm_step.label_col  = sf::Color(8, 120, 80);
    sm_step.is_bars    = true;
    sm_step.bars       = std::vector<float>(probs.data.begin(), probs.data.end());
    sm_step.bars_title = "probabilities";
    res.steps.push_back(std::move(sm_step));

    return res;
}

static sf::Text make_text(const std::string& s, const sf::Font& f,
                           unsigned size, sf::Color c, float x, float y)
{
    sf::Text t(s, f, size);
    t.setFillColor(c);
    t.setPosition(x, y);
    return t;
}

static sf::Text make_bold(const std::string& s, const sf::Font& f,
                           unsigned size, sf::Color c, float x, float y)
{
    sf::Text t(s, f, size);
    t.setStyle(sf::Text::Bold);
    t.setFillColor(c);
    t.setPosition(x, y);
    return t;
}

static sf::Color plasma(float v) {
    v = std::clamp(v, 0.f, 1.f);
    const float r[] = { 0.06f, 0.55f, 0.88f, 0.18f, 0.95f };
    const float g[] = { 0.03f, 0.15f, 0.42f, 0.42f, 0.80f };
    const float b[] = { 0.04f, 0.08f, 0.05f, 0.72f, 0.55f };
    float idx = v * 4.f;
    int   i   = std::min((int)idx, 3);
    float t   = idx - i;
    return {
        (uint8_t)((r[i] + t*(r[i+1]-r[i])) * 255),
        (uint8_t)((g[i] + t*(g[i+1]-g[i])) * 255),
        (uint8_t)((b[i] + t*(b[i+1]-b[i])) * 255)
    };
}

static void draw_tile(sf::RenderTarget& win,
                       const float* data, int H, int W,
                       float px, float py, float tile_px)
{
    float mn  = *std::min_element(data, data + H*W);
    float mx  = *std::max_element(data, data + H*W);
    float rng = mx - mn + 1e-6f;
    float cs  = tile_px / W;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            float v = (data[r*W+c] - mn) / rng;
            sf::RectangleShape cell({cs + 0.5f, cs * ((float)H/W) + 0.5f});
            cell.setPosition(px + c*cs, py + r*(tile_px/W));
            cell.setFillColor(plasma(v));
            win.draw(cell);
        }
}

static float draw_filter_grid(sf::RenderTarget& win,
                               const Step& step,
                               float x0, float y0, float tile_px, float gap)
{
    int   max_col = std::max(1, (int)((PANEL_W - x0 - 10) / (tile_px + gap)));
    float x = x0, y = y0;

    for (int f = 0; f < step.filters; ++f) {
        draw_tile(win, step.data.data() + f * step.H * step.W,
                  step.H, step.W, x, y, tile_px);
        sf::RectangleShape border({tile_px, tile_px});
        border.setPosition(x, y);
        border.setFillColor(sf::Color::Transparent);
        border.setOutlineColor(sf::Color(200, 195, 185));
        border.setOutlineThickness(1);
        win.draw(border);
        x += tile_px + gap;
        if ((f + 1) % max_col == 0) { x = x0; y += tile_px + gap; }
    }
    if (step.filters % max_col != 0) y += tile_px + gap;
    return y - y0;
}

static float draw_pixel_row(sf::RenderTarget& win, const sf::Font& font,
                             const Step& step, int pred, float x0, float y0)
{
    const auto& v = step.bars;
    int   n     = (int)v.size();
    float mn    = *std::min_element(v.begin(), v.end());
    float mx    = *std::max_element(v.begin(), v.end());
    float rng   = mx - mn + 1e-6f;
    float avail = PANEL_W - x0 - 12.f;
    float cell_w = std::min(avail / n, 52.f);   // plus large pour les logits n=10
    float cell_h = cell_w * 1.2f;
    bool  is_sm  = (n == 10 && step.bars_title == "probabilities");

    for (int i = 0; i < n; ++i) {
        float norm = (v[i] - mn) / rng;
        sf::Color col = (is_sm && i == pred) ? COL_BAR : plasma(norm);
        sf::RectangleShape cell({cell_w - 1, cell_h});
        cell.setPosition(x0 + i * cell_w, y0);
        cell.setFillColor(col);
        win.draw(cell);
        if (n <= 10) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%d", i);
            sf::Text lbl(buf, font, 13);
            lbl.setFillColor(i == pred && is_sm ? COL_WHITE : COL_DIM);
            lbl.setPosition(x0 + i * cell_w + cell_w * 0.15f, y0 + cell_h + 2);
            win.draw(lbl);
            if (std::abs(v[i]) > 0.001f) {
                std::snprintf(buf, sizeof(buf), "%.1f", v[i]);
                sf::Text val(buf, font, 10);
                val.setFillColor(sf::Color(140, 140, 150));
                val.setPosition(x0 + i * cell_w, y0 + cell_h + 16);
                win.draw(val);
            }
        }
    }
    return cell_h + (n <= 10 ? 32.f : 4.f);
}

static float draw_softmax_cols(sf::RenderTarget& win, const sf::Font& font,
                                const Step& step, int pred, float x0, float y0)
{
    const auto& v = step.bars;
    const float MAX_H = 80.f;
    const float COL_W = 52.f;
    const float GAP   = 8.f;
    float baseline = y0 + MAX_H;

    for (int i = 0; i < 10; ++i) {
        float col_h = std::max(2.f, v[i] * MAX_H);
        float cx    = x0 + i * (COL_W + GAP);
        bool  best  = (i == pred);

        sf::RectangleShape bar({COL_W, col_h});
        bar.setPosition(cx, baseline - col_h);
        bar.setFillColor(best ? COL_BAR : plasma(v[i]));
        win.draw(bar);
        bar.setFillColor(sf::Color::Transparent);
        bar.setOutlineColor(sf::Color(200, 195, 185));
        bar.setOutlineThickness(1);
        win.draw(bar);

        char buf[8];
        std::snprintf(buf, sizeof(buf), "%d", i);
        sf::Text lbl(buf, font, 14);
        if (best) lbl.setStyle(sf::Text::Bold);
        lbl.setFillColor(best ? COL_WHITE : COL_DIM);
        lbl.setPosition(cx + COL_W * 0.28f, baseline + 3);
        win.draw(lbl);

        if (v[i] > 0.005f) {
            std::snprintf(buf, sizeof(buf), "%.0f%%", v[i] * 100.f);
            sf::Text pct(buf, font, 12);
            if (best) pct.setStyle(sf::Text::Bold);
            pct.setFillColor(best ? COL_GREEN : COL_DIM);
            pct.setPosition(cx + 2, baseline - col_h - 16);
            win.draw(pct);
        }
    }
    return MAX_H + 32.f;
}

int main(int argc, char* argv[])
{
    std::string weight_path = "weights.bin";
    if (argc >= 2) weight_path = argv[1];

    Sequential net = build_cnn();
    bool weights_ok = false;
    try {
        net.loadWeights(weight_path);
        weights_ok = true;
        std::cout << "Loaded: " << weight_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[WARN] " << e.what() << "\n";
    }

    sf::RenderWindow window(sf::VideoMode(WIN_W, WIN_H),
                             "CNN Visualiser - all steps");
    window.setFramerateLimit(60);

    sf::Font font;
    font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") ||
    font.loadFromFile("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf") ||
    font.loadFromFile("/System/Library/Fonts/Helvetica.ttc") ||
    font.loadFromFile("arial.ttf");

    float     pixels[GRID * GRID] = {};
    bool      drawing    = false;
    CNNResult result;
    bool      has_result = false;

    sf::RenderTexture panel_tex;
    panel_tex.create(PANEL_W, WIN_H);

    auto clearCanvas = [&]() {
        std::fill(pixels, pixels + GRID*GRID, 0.f);
        result = {}; has_result = false;
    };

    auto render_panel = [&]() {
        panel_tex.clear(BG_PANEL);
        float y   = 12.f;
        float PAD = 16.f;

        panel_tex.draw(make_bold("CNN Forward Pass - all steps",
                                  font, 22, COL_WHITE, PAD, y));
        y += 30;
        panel_tex.draw(make_text(
            weights_ok ? ("Weights: " + weight_path)
                       : "Weights: RANDOM - run ./mnist_cnn first",
            font, 14,
            weights_ok ? sf::Color(120, 115, 105) : sf::Color(210, 100, 30),
            PAD, y));
        y += 18;

        sf::RectangleShape hsep({(float)PANEL_W - 2*PAD, 1});
        hsep.setPosition(PAD, y);
        hsep.setFillColor(COL_GREY);
        panel_tex.draw(hsep);
        y += 10;

        if (!has_result) {
            panel_tex.draw(make_bold("Draw a digit ->",
                                      font, 18, COL_CYAN, PAD, y));
            y += 28;
            const char* arch[] = {
                "0  Conv2D   1->8,  k=3    [8,26,26]",
                "1  ReLU                   [8,26,26]",
                "2  MaxPool2D  2x2         [8,13,13]",
                "3  Conv2D   8->16, k=3  [16,11,11]",
                "4  ReLU                  [16,11,11]",
                "5  MaxPool2D  2x2        [16, 5, 5]",
                "6  Flatten                   [400]",
                "7  Linear  400->128          [128]",
                "8  ReLU                       [128]",
                "9  Linear  128->10             [10]",
            };
            for (auto* l : arch) {
                panel_tex.draw(make_text(l, font, 16, COL_CYAN, PAD + 8, y));
                y += 22;
            }
            panel_tex.display();
            return;
        }

        int n_fmap = 0, n_bars = 0;
        for (auto& s : result.steps)
            s.is_bars ? ++n_bars : ++n_fmap;

        float PRED_H   = 80.f;
        float avail_h  = WIN_H - y - PRED_H;
        float sep_h    = 8.f;
        float label_h  = 22.f;
        int   n_steps  = (int)result.steps.size();
        float per_step = avail_h / n_steps - sep_h - label_h;

        float tile_8  = std::min(per_step, (float)(PANEL_W - 2*PAD) / 8  - 4);
        float tile_16 = std::min(per_step, (float)(PANEL_W - 2*PAD) / 16 - 4);
        tile_8  = std::max(tile_8,  28.f);
        tile_16 = std::max(tile_16, 20.f);

        for (auto& step : result.steps) {
            panel_tex.draw(make_bold(step.label, font, 17,
                                      step.label_col, PAD, y));
            y += label_h;

            if (step.is_bars) {
                bool is_sm = (step.bars_title == "probabilities");
                float used = is_sm
                    ? draw_softmax_cols(panel_tex, font, step,
                                         result.prediction, PAD + 8, y)
                    : draw_pixel_row   (panel_tex, font, step,
                                         result.prediction, PAD + 8, y);
                y += used + 4;
            } else {
                float tile = (step.filters <= 8) ? tile_8 : tile_16;
                float gap  = 4.f;
                float used = draw_filter_grid(panel_tex, step, PAD, y, tile, gap);
                y += used + 4;
            }

            sf::RectangleShape sep2({(float)PANEL_W - 2*PAD, 1});
            sep2.setPosition(PAD, y);
            sep2.setFillColor(COL_GREY);
            panel_tex.draw(sep2);
            y += sep_h;
        }

        if (result.prediction >= 0) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "Prediction: %d   (%.0f%%)",
                          result.prediction,
                          result.scores[result.prediction] * 100.f);
            panel_tex.draw(make_bold(buf, font, 28, COL_RED, PAD, y + 12));
        }

        panel_tex.display();
    };

    while (window.isOpen())
    {
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed)   window.close();
            if (ev.type == sf::Event::KeyPressed) {
                if (ev.key.code == sf::Keyboard::Escape) window.close();
                if (ev.key.code == sf::Keyboard::C)      clearCanvas();
            }
            if (ev.type == sf::Event::MouseButtonPressed &&
                ev.mouseButton.button == sf::Mouse::Left)
                drawing = true;
            if (ev.type == sf::Event::MouseButtonReleased &&
                ev.mouseButton.button == sf::Mouse::Left)
                drawing = false;
        }

        if (drawing) {
            auto m  = sf::Mouse::getPosition(window);
            int  lx = m.x - PANEL_W;
            int  gx = lx / CELL, gy = m.y / CELL;
            for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                int nx = gx+dx, ny = gy+dy;
                if (nx>=0 && nx<GRID && ny>=0 && ny<GRID)
                    pixels[ny*GRID+nx] = std::min(1.f, pixels[ny*GRID+nx] + 0.6f);
            }
            result     = run_cnn(net, pixels);
            has_result = true;
        }

        render_panel();
        window.clear(BG_DARK);

        sf::Sprite panel_sprite(panel_tex.getTexture());
        panel_sprite.setPosition(0, 0);
        window.draw(panel_sprite);

        sf::RectangleShape sep({2.f, (float)WIN_H});
        sep.setPosition(PANEL_W, 0);
        sep.setFillColor(COL_SEP);
        window.draw(sep);

        {
            auto lbl = make_bold("Input  [1 x 28 x 28]",
                                  font, 18, COL_WHITE, PANEL_W + 10, 8);
            window.draw(lbl);
            window.draw(make_text("[C] clear   [Esc] quit",
                                   font, 13, COL_DIM, PANEL_W + 10, WIN_H - 20));
        }

        for (int row = 0; row < GRID; ++row)
        for (int col = 0; col < GRID; ++col) {
            float   v = pixels[row*GRID+col];
            uint8_t c = (uint8_t)((1.f - v) * 255);
            sf::RectangleShape cell({(float)(CELL-1), (float)(CELL-1)});
            cell.setPosition(PANEL_W + col*CELL, row*CELL);
            cell.setFillColor(sf::Color(c, c, c));
            window.draw(cell);
        }

        for (int i = 0; i <= GRID; ++i) {
            sf::RectangleShape lh({(float)DRAW_W, 1.f});
            lh.setPosition(PANEL_W, i*CELL);
            lh.setFillColor(sf::Color(220, 218, 212));
            window.draw(lh);
            sf::RectangleShape lv({1.f, (float)WIN_H});
            lv.setPosition(PANEL_W + i*CELL, 0);
            lv.setFillColor(sf::Color(220, 218, 212));
            window.draw(lv);
        }

        sf::RectangleShape border({(float)(DRAW_W-2), (float)(WIN_H-2)});
        border.setPosition(PANEL_W+1, 1);
        border.setFillColor(sf::Color::Transparent);
        border.setOutlineColor(COL_SEP);
        border.setOutlineThickness(3);
        window.draw(border);

        window.display();
    }

    return 0;
}