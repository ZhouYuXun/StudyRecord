import Item from "./Item";

const List = ({ listData, deleteData }) => {
  console.log("listData", listData);
  return (
    <div className="list">
      {
        listData.map((item) => {
        const { id, note, date, time } = item;
        return <Item key={id} id={id} note={note} date={date} time={time} deleteData={deleteData}/>;
      })}
    </div>
  );
};

export default List;
