import logging
import flwr as fl

class TimeoutClientManager(fl.server.SimpleClientManager):
    """
    ClientManager que espera un tiempo máximo fijo (wait_timeout, en segundos)
    a que se conecten los clientes esperados antes de la primera ronda.

    - Si todos los clientes esperados se conectan dentro de ese plazo, sigue
      exactamente igual que el comportamiento por defecto de Flower.
    - Si se agota el plazo y hay algunos clientes conectados, continúa el
      entrenamiento únicamente con los clientes disponibles.
    - Si se agota el plazo y no hay ningún cliente conectado, detiene el
      entrenamiento con un error claro.
    - Esta espera solo ocurre una vez (antes de la primera ronda). En rondas
      posteriores no se vuelve a bloquear.
    """

    def __init__(self, expected_clients: int, wait_timeout: float):
        super().__init__()
        self.expected_clients = expected_clients
        self.wait_timeout = wait_timeout
        self._initial_wait_done = False

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        if self._initial_wait_done:
            # La espera inicial ya se resolvió; no volver a bloquear en
            # rondas siguientes.
            return True

        with self._cv:
            success = self._cv.wait_for(
                lambda: len(self.clients) >= num_clients,
                timeout=self.wait_timeout,
            )

        connected = len(self.clients)

        if success:
            logging.info(
                "All expected clients (%d) connected within the %.0f s timeout.",
                num_clients,
                self.wait_timeout,
            )

        elif connected == 0:
            logging.error(
                "Timeout of %.0f s reached: no clients are connected. "
                "Stopping training.",
                self.wait_timeout,
            )

            raise RuntimeError(
                f"No clients connected within the {self.wait_timeout:.0f} "
                "second timeout. Training cannot continue."
            )

        else:
            logging.warning(
                "Timeout of %.0f s reached: %d/%d clients connected. "
                "Continuing training with only the available %d client(s).",
                self.wait_timeout,
                connected,
                num_clients,
                connected,
            )

        self._initial_wait_done = True

        # Siempre devolvemos True: el manejo de "menos clientes de los
        # solicitados" se hace en sample(), en vez de fallar aquí.
        return True

    def sample(self, num_clients, min_num_clients=None, criterion=None):
        if min_num_clients is None:
            min_num_clients = num_clients

        self.wait_for(min_num_clients)

        available_cids = list(self.clients)

        if criterion is not None:
            available_cids = [
                cid
                for cid in available_cids
                if criterion.select(self.clients[cid])
            ]

        if not available_cids:
            logging.error(
                "No clients are available to sample. Stopping training."
            )

            raise RuntimeError(
                "No clients are available. Training cannot continue."
            )

        # En vez de fallar cuando hay menos clientes conectados de los
        # solicitados, se ajusta el tamaño de muestra a lo disponible.
        actual_num_clients = min(num_clients, len(available_cids))

        if actual_num_clients < num_clients:
            logging.info(
                "Clients requested %d, but only %d are available. "
                "Continuing with %d clients.",
                num_clients,
                len(available_cids),
                actual_num_clients,
            )

        sampled_cids = random.sample(available_cids, actual_num_clients)

        return [self.clients[cid] for cid in sampled_cids]
