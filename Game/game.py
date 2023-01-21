import pygame as pg    # noqa


class Game:

    def __init__(self) -> None:
        if not pg.get_init():
            pg.init()

        # Main window
        self.title: str = 'PyGameWindow'
        self.size: tuple = (640, 480)
        self.window_flags: int = 0
        self.fps: int = 60
        self.running: bool = True
        self.rendering: bool = True
        self.clock = pg.time.Clock()

        # Event
        self.mouse_x = 0
        self.mouse_y = 0
        self.keys = []

        # Render
        # User sprite list - TODO
        self.sprites: list[pg.Rect] = []
        # User custom render callbacks list - TODO
        self.window = None

    def __del__(self):
        pg.quit()

    def loop_forever(self) -> None:
        self.__setup()
        while self.running:
            self.__eventHandler()
            self.loop()
            self.__render()

    def loop_once(self) -> bool:
        self.__eventHandler()
        self.loop()
        self.__render()
        return self.running

    def loop(self) -> None:
        """ User should override this method """
        ...

    def __setup(self):
        self.set_title(self.title)
        self.set_window()
        self.setup()

    def setup(self) -> None:
        """ User should override this method """
        ...

    def __eventHandler(self):
        self.mouse_x, self.mouse_y = pg.mouse.get_pos()
        self.keys = pg.key.get_pressed()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
                break
            else:
                self.onEvent(event)

    def onEvent(self, event) -> None:
        """ User should override this method """
        ...

    def __render(self):
        if self.rendering and self.running:
            self.onRender()
            pg.display.flip()
            self.clock.tick(self.fps)

    def onRender(self) -> None:
        """ User should override this method """
        ...

    def set_window(self) -> None:
        """ Avoid calling outside of PyGameBase instance """
        if not self.window:
            self.window = pg.display.get_surface()
            if self.window is None:
                self.window = pg.display.set_mode(self.size,
                                                  self.window_flags)

    def get_window(self) -> pg.Surface:
        return pg.display.get_surface()

    @staticmethod
    def set_title(title: str) -> None:
        pg.display.set_caption(title)
